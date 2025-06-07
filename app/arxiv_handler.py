import arxiv
import requests
import os

# def fetch_arxiv_pdf(query, save_dir="data/"):
#     search = arxiv.Search(query=query, max_results=1)
#     result = next(search.results())
#     pdf_url = result.pdf_url
#     paper_id = result.entry_id.split("/")[-1]
#     save_path = os.path.join(save_dir, f"{paper_id}.pdf")

#     # Download and save
#     response = requests.get(pdf_url)
#     with open(save_path, 'wb') as f:
#         f.write(response.content)

#     return save_path, result.title


def fetch_arxiv_pdfs(query, max_results=3, save_dir="data/"):
    search = arxiv.Search(query=query, max_results=max_results)
    papers = []
    for result in search.results():
        title = result.title
        pdf_url = result.pdf_url
        paper_id = result.entry_id.split("/")[-1]
        save_path = os.path.join(save_dir, f"{paper_id}.pdf")

        response = requests.get(pdf_url)
        with open(save_path, 'wb') as f:
            f.write(response.content)

        papers.append((save_path, title, paper_id))
    return papers


