"""Parse the one-to-many Entrez GeneID field in UniProt idmapping."""
GENE_PREFIX='http://mowl.borg/'

def entrez_ids(value):
    text=str(value)
    if text.startswith(GENE_PREFIX):text=text[len(GENE_PREFIX):]
    fields=[part.strip() for part in text.split(';')]
    if not fields or any(not part.isdecimal() or int(part)<=0 for part in fields):
        raise ValueError(f'Invalid Entrez mapping: {value!r}')
    return sorted({str(int(part)) for part in fields},key=int)

def gene_iris(value):
    return [GENE_PREFIX+i for i in entrez_ids(value)]
