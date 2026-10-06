.PHONY: verify test

verify:
	python -m scripts.verify --profile release --output _evidence/verify

test:
	python -m scripts.verify --profile unit --output _evidence/unit
