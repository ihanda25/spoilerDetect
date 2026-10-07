"""Collect public episode references/plots for seven pilot series. No model calls."""
import argparse,hashlib,json,re,time
from pathlib import Path
from urllib.parse import urljoin,urlparse
import requests
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/episode-context';OUT.mkdir(parents=True,exist_ok=True)
REPORT=ROOT/'reports/context-training-data-audit'
SOURCES={p:'https://en.wikipedia.org/wiki/'+slug for p,slug in {
'Fringe':'List_of_Fringe_episodes','Sherlock':'List_of_Sherlock_episodes','Firefly':'List_of_Firefly_episodes','PersonOfInterest':'List_of_Person_of_Interest_episodes','Homeland':'List_of_Homeland_episodes','Nikita':'List_of_Nikita_episodes','LifeOnMars2006':'List_of_Life_on_Mars_(British_TV_series)_episodes'}.items()}
SEASON_PATTERNS={'Fringe':r'/Fringe_season_\d+$','PersonOfInterest':r'/Person_of_Interest_season_\d+$','Homeland':r'/Homeland_season_\d+$','Nikita':r'/Nikita_season_\d+$'}
def norm(s):return re.sub(r'[^a-z0-9]','',s.lower())
def clean(node):
 copy=BeautifulSoup(str(node),'html.parser')
 for x in copy.select('sup.reference, .noprint, style, script'):x.decompose()
 return ' '.join(copy.get_text(' ',strip=True).split())

class Fetcher:
 def __init__(self,offline=False):self.previous=0;self.sources={};self.offline=offline
 def get(self,url):
  assert urlparse(url).hostname=='en.wikipedia.org'
  path=OUT/(hashlib.sha256(url.encode()).hexdigest()+'.html')
  meta=path.with_suffix('.json')
  if not path.exists():
   if self.offline:raise FileNotFoundError('Missing cached public source: '+url)
   time.sleep(max(0,1-(time.monotonic()-self.previous)))
   response=requests.get(url,headers={'User-Agent':'SpoilerDetectResearch/0.1 (local dataset preparation; Python requests)'},timeout=30);self.previous=time.monotonic();response.raise_for_status()
   path.write_bytes(response.content)
   meta.write_text(json.dumps({'requested_url':url,'resolved_url':response.url,'fetched_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sha256':hashlib.sha256(response.content).hexdigest()},indent=2)+'\n')
  data=json.loads(meta.read_text());self.sources[url]=data
  return BeautifulSoup(path.read_bytes(),'html.parser')

def episodes(soup,page,url):
 result=[]
 for row in soup.select('tr.vevent'):
  titlecell=row.select_one('.summary')
  if not titlecell:continue
  title_text=clean(titlecell)
  titles=[a.get_text(' ',strip=True) for a in titlecell.select('a') if not a.find_parent('sup') and not re.match(r'^\[',a.get_text())]
  titles=list(dict.fromkeys(a.strip() for a in titles+re.findall(r'"([^"]+)"',title_text) if a.strip()))
  title=title_text.strip('"“” ')
  number=row.select_one('th[id]');anchor=number.get('id') if number else None
  nxt=row.find_next_sibling('tr');plot=''
  if nxt and 'expand-child' in nxt.get('class',[]):plot=clean(nxt)
  href=next((a.get('href') for a in titlecell.select('a') if '/wiki/' in a.get('href','')),None)
  link=urljoin(url,href) if href else None
  heading=row.find_previous(['h2','h3']);season=clean(heading) if heading else None
  result.append({'work_page':page,'episode_title':title,'title_aliases':titles or [title],'episode_url':link,'episode_list_url':url,'episode_anchor':anchor,'season_heading':season,'plot':plot,'plot_source_url':url+'#'+anchor if plot and anchor else url if plot else None,'context_kind':'episode_plot','source_license_url':'https://en.wikipedia.org/wiki/Wikipedia:Copyrights'})
 return result

def article_plot(soup,part=None):
 header=next((h for h in soup.find_all(['h2','h3']) if clean(h).lower() in ['plot','plot summary','synopsis']),None)
 if header is None:return ''
 container=header.parent if 'mw-heading' in header.parent.get('class',[]) else header
 parts=[];active=part is None
 level=int(header.name[1])
 def section_nodes(nodes):
  for node in nodes:
   if getattr(node,'name',None)=='section':yield from section_nodes(node.children)
   else:yield node
 for nxt in section_nodes(container.next_siblings):
  if not getattr(nxt,'name',None):continue
  heading=nxt if nxt.name in ['h2','h3','h4'] else nxt.find(['h2','h3','h4']) if 'mw-heading' in nxt.get('class',[]) else None
  if heading is not None:
   if int(heading.name[1])<=level:break
   if part is not None:
    if re.search(r'part\s*(?:'+str(part)+'|'+{1:'one',2:'two'}.get(part,str(part))+r')\b',clean(heading),re.I):active=True
    elif active:break
   continue
  if active and nxt.name in ['p','ul','ol']:parts.append(clean(nxt))
 return '\n'.join(p for p in parts if p)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');args=parser.parse_args()
 fetch=Fetcher(offline=args.offline);all_episodes=[]
 for page,url in SOURCES.items():
  soup=fetch.get(url);records=episodes(soup,page,url)
  season_urls=set()
  for a in soup.select('a[href]'):
   href=urljoin(url,a['href'])
   if page in SEASON_PATTERNS and re.search(SEASON_PATTERNS[page],urlparse(href).path):season_urls.add(href)
  for season_url in sorted(season_urls):
   season_soup=fetch.get(season_url)
   # Prefer season-table plot rows over metadata-only series tables.
   extra=episodes(season_soup,page,season_url)
   indexed={norm(r['episode_title']):r for r in records}
   for r in extra:
    key=norm(r['episode_title'])
    if key not in indexed or r['plot']:indexed[key]=r
   records=list(indexed.values())
  all_episodes.extend(records)
  print(page,'episodes',len(records),'plots',sum(bool(r['plot']) for r in records),'season pages',len(season_urls),flush=True)
 # Fetch detailed plots for explicitly named episodes in the fixed pilot/quality sample.
 from add_episode_context import explicit_episodes
 targets=[]
 for name in ['train.jsonl','validation.jsonl','quality-review.jsonl']:
  targets += [json.loads(l) for l in (ROOT/'data/processed/sentence-context-pilot'/name).read_text().splitlines()]
 for record in all_episodes:
  if not record['episode_url']:continue
  explicit=any(r['work_page']==record['work_page'] and record['episode_title'] in explicit_episodes(r['text'],[e for e in all_episodes if e['work_page']==record['work_page']]) for r in targets)
  if not explicit:continue
  soup=fetch.get(record['episode_url']);part=re.search(r'Part (\d+)',record['episode_title']);plot=article_plot(soup,int(part.group(1)) if part else None)
  if plot:record.update(list_plot=record['plot'],plot=plot,plot_source_url=record['episode_url']+'#Plot',context_kind='detailed_episode_plot')
  print('Episode article',record['work_page'],record['episode_title'],'plot chars',len(plot),flush=True)
 (OUT/'episodes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_episodes))
 manifest={'sources':fetch.sources,'episodes':len(all_episodes),'episodes_with_plot':sum(bool(r['plot']) for r in all_episodes),'detailed_episode_plots':sum(r['context_kind']=='detailed_episode_plot' for r in all_episodes),'by_work':{page:{'episodes':sum(r['work_page']==page for r in all_episodes),'with_plot':sum(r['work_page']==page and bool(r['plot']) for r in all_episodes)} for page in SOURCES},'notes':['Public plot text stored locally under gitignored data/raw with attribution/source hashes','Episode metadata/plots collected independently of spoiler labels','All seasons included; no viewer-progress personalization','No model inference or training','Raw source text is not redistributed in committed reports']}
 (REPORT/'episode-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
