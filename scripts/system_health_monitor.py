"""
Background system health monitor — polls RAM, CPU, GPU (via nvidia-smi if
present, else best-effort), and Windows System Event Log entries relevant to
unexpected shutdowns, appending one JSON line per sample.

WHY THIS EXISTS: two unexpected shutdowns (Event ID 6008) happened today
during sustained heavy compute (transformer inference + Ollama GPU load),
with no BSOD/bugcheck/thermal-trip event logged — consistent with a hard
power loss rather than a software crash. This can't fix a hardware/power
issue, but it gives a timestamped trail of resource pressure leading up to
any future interruption, which a bare "the process died" does not.

Usage:
    python -m scripts.system_health_monitor --interval 15
"""
import argparse
import json
import os
import subprocess
import time

OUT_FILE = os.path.join("_evidence", "system_health.jsonl")


def sample_ram():
    import psutil
    vm = psutil.virtual_memory()
    return {
        "ram_available_gb": round(vm.available / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_percent_used": vm.percent,
    }


def sample_cpu():
    import psutil
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "cpu_freq_mhz": (psutil.cpu_freq().current if psutil.cpu_freq() else None),
    }


def sample_gpu():
    """
    No nvidia-smi on this AMD machine (confirmed earlier this session), and
    no rocm-smi either. Ollama's own /api/ps reports VRAM residency for
    whatever model is currently loaded, which is the closest available
    signal without vendor tooling.
    """
    try:
        import requests
        r = requests.get("http://localhost:11434/api/ps", timeout=2)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return {"ollama_reachable": True,
                    "models_resident": [
                        {"name": m["name"], "size_vram_gb": round(m.get("size_vram", 0) / 1e9, 2)}
                        for m in models
                    ]}
        return {"ollama_reachable": False, "http_status": r.status_code}
    except Exception:
        return {"ollama_reachable": False}


def sample_temperature():
    """
    ACPI thermal zone temperature via the Windows performance counter
    `\\Thermal Zone Information(*)\\Temperature` — works without admin
    rights, unlike the WMI MSAcpi_ThermalZoneTemperature class (confirmed
    "Access denied" on this machine). Reports the highest of any zone found,
    since a single ACPI thermal zone on a laptop typically tracks whatever
    is running hottest (often near the CPU package), and the safety
    decision that consumes this value only cares about the worst case.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Thermal Zone Information(*)\\Temperature' "
             "-ErrorAction Stop).CounterSamples | "
             "ForEach-Object { $_.CookedValue }"],
            capture_output=True, text=True, timeout=10,
        )
        values_kelvin = [float(v) for v in out.stdout.split() if v.strip()]
        if not values_kelvin:
            return {"available": False, "celsius": None}
        celsius = [round(v - 273.15, 1) for v in values_kelvin]
        return {"available": True, "celsius": max(celsius), "zones_celsius": celsius}
    except Exception as e:
        return {"available": False, "celsius": None, "error": str(e)}


def sample_recent_shutdown_events():
    """Checks for a NEW unexpected-shutdown event (6008) since the monitor
    started, using PowerShell (no admin rights needed for the System log)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-WinEvent -FilterHashtable @{LogName='System'; Id=6008; "
             "StartTime=(Get-Date).AddMinutes(-2)} -MaxEvents 1 -ErrorAction "
             "SilentlyContinue | Select-Object -ExpandProperty TimeCreated"],
            capture_output=True, text=True, timeout=10,
        )
        hit = out.stdout.strip()
        return {"unexpected_shutdown_in_last_2min": bool(hit), "detail": hit or None}
    except Exception as e:
        return {"unexpected_shutdown_in_last_2min": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    print(f"System health monitor starting, sampling every {args.interval}s -> {OUT_FILE}")

    while True:
        sample = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        sample.update(sample_ram())
        sample.update(sample_cpu())
        sample["gpu"] = sample_gpu()
        sample["temperature"] = sample_temperature()
        sample["shutdown_check"] = sample_recent_shutdown_events()

        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")

        # Flag loudly in stdout (captured to the log file the wrapper redirects
        # to) whenever RAM crosses into the zone that has preceded every OOM
        # kill and stall observed this session.
        if sample["ram_available_gb"] < 1.0:
            print(f"[{sample['timestamp']}] LOW RAM WARNING: "
                  f"{sample['ram_available_gb']}GB available")

        temp_c = sample["temperature"].get("celsius")
        if temp_c is not None and temp_c > 65:
            print(f"[{sample['timestamp']}] HIGH TEMPERATURE WARNING: {temp_c}C")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
