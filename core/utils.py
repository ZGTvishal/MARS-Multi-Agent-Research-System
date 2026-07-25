import numpy as np

def get_paper_by_id(papers: list[dict], entry_id: str) -> dict:
    """
    Generates a dict of with matching url and entry_id

        Args: 
            List of paper dics and entry_id

        Returns: 
            dict of matching papers
        
        Raises:
            Value error when no entry_id is matched
    """
    try:
        return next(p for p in papers if p["url"] == entry_id)
    except StopIteration:
        raise ValueError(f'No paper found with entry_id - {entry_id}')


def get_index_path(entry_id: str) -> str:
    pass




def split_text(text: str) -> list[str]:
    """
    Generates a list of chunks (5 elements) from a string with overlap of 3 words from the previous element excluding the 0th element. 
        
        Args:
            str: The input abstract.
        
        Returns: 
            List (5 element) of chunks with 3 word overlap
        
    """
    words = text.split()

    if len(words) <= 20:
        return [text]
    N = 5
    chunks = np.array_split(words, N)
    overlapped_chunks = []
    for i in range(len(chunks)):
        current = chunks[i].tolist()

        # Previous overlap
        if i > 0:
            previous = chunks[i - 1][-3:].tolist()
        else:
            previous = []

        overlapped = previous + current
        overlapped_chunks.append(" ".join(overlapped))

    return overlapped_chunks