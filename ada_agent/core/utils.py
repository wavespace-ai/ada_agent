def parse_frontmatter(content: str):
    """
    Parses YAML frontmatter from a string.
    Returns (frontmatter_dict, remaining_content).
    """
    if not content.startswith("---"):
        return {}, content
    
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    
    # Simple YAML parser (avoiding PyYAML dependency if possible, but PyYAML is safer)
    # Since we didn't add PyYAML to requirements, let's do a basic parse or add it.
    # Actually, for robustness, let's assume we can add PyYAML or do simple line parsing.
    # Let's try simple line parsing for now to minimize dependencies, or better, 
    # since we have 'requirements.txt', let's add pyyaml to it in the next step if needed.
    # For now, I'll implement a basic key-value parser.
    
    frontmatter_raw = parts[1]
    body = parts[2]
    
    metadata = {}
    for line in frontmatter_raw.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
            
    return metadata, body.strip()
