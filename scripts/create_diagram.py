from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.compute import Server
from diagrams.onprem.database import Cassandra
from diagrams.onprem.security import Vault
from diagrams.onprem.monitoring import Splunk
from diagrams.programming.flowchart import Decision
import os

# Fix for Graphviz path on Windows
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Common graph attributes
graph_attr = {
    "fontsize": "24",
    "bgcolor": "white"
}

with Diagram("Gatekeeper Architecture", show=False, graph_attr=graph_attr, direction="TB"):
    
    user = Custom("User Input", "./assets/user_icon.png") # Or use a generic node if no icon

    with Cluster("Layer 1: Input Rail (Privacy)"):
        pii = Vault("PII Redaction\n(Regex + NER)")
        norm = Server("Multi-Modal\nNormalization")
        
    with Cluster("Layer 2: Neuro-Symbolic Core"):
        regex = Decision("Symbolic Veto\n(Jailbreak Regex)")
        vector = Cassandra("Vector DB\n(Threat Anchors)")
        domain = Decision("Domain Gate")
        
    with Cluster("Layer 3: Execution Rail"):
        judge = Server("Semantic Judge\n(Mistral-7B)")
        arbiter = Custom("Policy Arbiter", "./assets/lock.png")
        audit = Splunk("Audit Log\n(JSONL)")

    llm = Server("Local LLM")
    block = Custom("BLOCK", "./assets/stop.png")

    # Flow
    user >> norm >> pii >> regex
    
    # Symbolic Fail
    regex >> Edge(label="Match") >> block
    
    # Symbolic Pass
    regex >> Edge(label="Clean") >> domain
    
    # Domain Check
    domain >> Edge(label="Off-Topic") >> block
    domain >> Edge(label="Aligned") >> vector
    
    # Vector Check
    vector >> Edge(label="High Risk") >> block
    vector >> Edge(label="Ambiguous") >> judge
    vector >> Edge(label="Low Risk") >> arbiter
    
    # Judge
    judge >> Edge(label="Verdict") >> arbiter
    
    # Final
    arbiter >> audit >> llm