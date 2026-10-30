"""Validate a complete installed subset and its public archive cache before resume."""
import hashlib,json

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def verify_prefix(workspace,cache,manifest,verify_workspace):
 assert workspace.is_dir() and cache.is_dir()
 assert not workspace.is_symlink() and not cache.is_symlink()
 components={c['name']:c for c in manifest['components']}
 archives={c['archive']:c for c in manifest['components']}
 assert len(components)==len(archives)==len(manifest['components'])
 cached={}
 for path in cache.iterdir():
  assert path.is_file() and not path.is_symlink() and path.name in archives,path.name
  c=archives[path.name]
  assert path.stat().st_size==c['bytes'] and sha(path)==c['sha256'],path.name
  cached[c['name']]=c['sha256']
 sidecars=list((workspace/'MANIFESTS').iterdir())
 installed={}
 for path in sidecars:
  assert path.is_file() and not path.is_symlink() and path.suffix=='.json'
  name=path.stem;assert name in components and name in cached and name not in installed
  installed[name]=components[name]
 expected={}
 for c in installed.values():
  assert not(set(expected)&set(c['files']))
  expected.update({n:{k:r[k] for k in ('sha256','bytes')} for n,r in c['files'].items()})
 verify_workspace(workspace,expected,installed)
 return dict(status='passed',cached_archives=len(cached),installed_components=len(installed),installed_files=len(expected),cache_sha256=cached)
