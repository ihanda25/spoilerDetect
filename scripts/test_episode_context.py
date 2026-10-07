"""Regression checks for episode matching, parsing and label-blind retrieval."""
import unittest
from bs4 import BeautifulSoup
from collect_episode_context import episodes,article_plot
from add_episode_context import explicit_episodes,make_index,retrieve

class EpisodeContextTests(unittest.TestCase):
 def test_comma_short_title_and_part(self):
  eps=[{'episode_title':'6:02 AM EST','title_aliases':['6:02 AM EST']},{'episode_title':'Over There (Part 1)','title_aliases':['Over There (Part 1)']},{'episode_title':'Over There (Part 2)','title_aliases':['Over There (Part 2)']}]
  self.assertEqual(explicit_episodes('In "6:02 AM," an event happens.',eps),{'6:02 AM EST'})
  self.assertEqual(explicit_episodes('" In "6:02 AM," something happens.',eps),{'6:02 AM EST'})
  self.assertEqual(explicit_episodes('As of "Over There" part 1, we know it.',eps),{'Over There (Part 1)'})
  self.assertEqual(explicit_episodes('In "Over There", something happens.',eps),set())
 def test_parser_does_not_treat_metadata_as_plot_or_citation_as_title(self):
  soup=BeautifulSoup('<table><tr class="vevent"><th id="ep1">1</th><td class="summary">"<a href="/wiki/Test_episode">Alpha</a>"<sup class="reference"><a>[3]</a></sup></td></tr><tr class="expand-child"><td>A secret is revealed.<sup class="reference">[4]</sup></td></tr><tr class="vevent"><td class="summary">"Beta"</td></tr></table>','html.parser')
  rs=episodes(soup,'A','https://en.wikipedia.org/wiki/Example')
  self.assertEqual(rs[0]['plot'],'A secret is revealed.')
  self.assertEqual(rs[0]['title_aliases'],['Alpha'])
  self.assertEqual(rs[1]['plot'],'')
 def test_article_plot_excludes_production_section(self):
  soup=BeautifulSoup('<div class="mw-heading"><h2>Plot</h2></div><p>A revelation.</p><div class="mw-heading"><h2>Production</h2></div><p>Actor trivia.</p>','html.parser')
  self.assertEqual(article_plot(soup),'A revelation.')
 def test_multipart_plot(self):
  soup=BeautifulSoup('<h2>Plot</h2><h3>Part 1</h3><p>First event.</p><h3>Part 2</h3><p>Second event.</p><h2>Production</h2>','html.parser')
  self.assertEqual(article_plot(soup,1),'First event.')
  soup=BeautifulSoup('<div class="mw-heading"><h2>Plot</h2></div><section><h3>Part one</h3><p>First event.</p></section><section><h3>Part two</h3><p>Second event.</p></section><h2>Production</h2><p>Trivia.</p>','html.parser')
  self.assertEqual(article_plot(soup,1),'First event.')
  self.assertEqual(article_plot(soup,2),'Second event.')
  self.assertEqual(article_plot(soup,2),'Second event.')
  soup=BeautifulSoup(str(soup).replace('Part 1','Part one').replace('Part 2','Part two'),'html.parser')
  self.assertEqual(article_plot(soup,1),'First event.')
 def test_within_work_and_explicit_priority(self):
  eps=[{'work_page':'A','episode_title':'Alpha','title_aliases':['Alpha'],'plot':'The agent reveals a secret.'},{'work_page':'A','episode_title':'Beta','title_aliases':['Beta'],'plot':'The agent reveals a secret secret secret.'},{'work_page':'B','episode_title':'Alpha','title_aliases':['Alpha'],'plot':'The agent reveals a secret.'}]
  index=make_index(eps)
  rs,method=retrieve('In "Alpha," the agent reveals a secret.','A',index,eps)
  self.assertEqual(method,'explicit_episode_title')
  self.assertTrue(rs)
  self.assertTrue(all(d['episode']['work_page']=='A' and d['episode']['episode_title']=='Alpha' for _,d in rs))
  self.assertEqual(retrieve('zzzzzzzzz','A',index,eps)[0],[])

if __name__=='__main__':unittest.main()
