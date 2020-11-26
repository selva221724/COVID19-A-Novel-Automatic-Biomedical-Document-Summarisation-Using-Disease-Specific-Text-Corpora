from pymed import PubMed
pubmed = PubMed(tool="MyTool", email="selva221724@gmai.com")
results = pubmed.query("diabetes", max_results=10)
