import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dic = dict()
    for key in corpus:
            dic[key] = (1-damping_factor)/len(corpus.keys())
 
    links = corpus[page]
    
    if len(links) == 0:
        for key in dic.keys():
            dic[key] += damping_factor/len(corpus.keys())
        return dic
    
    for link in links:
        dic[link] += damping_factor/len(links)
 
    return dic
        


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    PR = dict()
    page = random.choice(list(corpus.keys()))

    for key in corpus.keys():
        PR[key] = 0
    
    while n > 0:
        n -= 1
        #count the times
        PR[page] += 1
        rand = random.random()
        trans = transition_model(corpus, page, damping_factor)

        #surf to next page, based on previous sample's transition model
        for key in trans.keys():
            if trans[key] < rand:
                rand -= trans[key]
            else:
                page = key
                break
    
    for key in PR.keys():
        #change the count times to probability by divide it by SAMPLES
        PR[key] = round(PR[key]/SAMPLES, 5)
    
    return PR
    
    


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    PR = dict()
    for key in corpus:
        PR[key] = 1/len(corpus)
    
    check = True
    while check:
        temp = {}

        for key in PR.keys():
            t = PR[key]

            # this line equal to (1- damping factor / N)
            temp[key] = (1-damping_factor)/len(corpus.keys())

            #iterate through every page links to the "key" page.
            for page, links in corpus.items():
                if key in links:
                    temp[key] += damping_factor * (PR[page]/len(links))

            #in order to check whether break the while loop.
            if abs(t - temp[key]) < 0.001:
                check = False
        #update this dictionary        
        PR.update(temp)
    
    for key in PR.keys():
        PR[key] = round(PR[key], 5)
        
    return PR       



if __name__ == "__main__":
    main()
