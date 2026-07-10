FORBIDDEN_PATTERNS = [
    {
        "name": "os",
        "tokens": ["import os", "from os", "os."],
        "risk_level": "High",
    },
    {
        "name": "subprocess",
        "tokens": ["subprocess"],
        "risk_level": "High",
    },
    {
        "name": "socket",
        "tokens": ["socket"],
        "risk_level": "High",
    },
    {
        "name": "requests",
        "tokens": ["requests"],
        "risk_level": "High",
    },
    {
        "name": "shutil",
        "tokens": ["shutil"],
        "risk_level": "High",
    },
    {
        "name": "open(",
        "tokens": ["open("],
        "risk_level": "High",
    },
    {
        "name": "eval(",
        "tokens": ["eval("],
        "risk_level": "High",
    },
    {
        "name": "exec(",
        "tokens": ["exec("],
        "risk_level": "High",
    },
    {
        "name": "compile(",
        "tokens": ["compile("],
        "risk_level": "High",
    },
    {
        "name": "pd.read_csv(",
        "tokens": ["pd.read_csv("],
        "risk_level": "Medium",
    },
    {
        "name": "to_csv(",
        "tokens": ["to_csv("],
        "risk_level": "Medium",
    },
    {
        "name": "to_excel(",
        "tokens": ["to_excel("],
        "risk_level": "Medium",
    },
]


def review_code_safety(code: str) -> dict:
    """Review generated code with simple string checks only."""
    normalized_code = code.lower()
    issues = []
    risk_levels = []

    for pattern in FORBIDDEN_PATTERNS:
        if any(token in normalized_code for token in pattern["tokens"]):
            issues.append(f'Disallowed pattern detected: {pattern["name"]}')
            risk_levels.append(pattern["risk_level"])

    if "High" in risk_levels:
        risk_level = "High"
    elif "Medium" in risk_levels:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "safe": not issues,
        "risk_level": risk_level,
        "issues": issues,
    }
