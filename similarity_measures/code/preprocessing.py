from pathlib import Path
import pickle as pkl
import re

DATA_D = Path(__file__).parent / "data"
RAW_DATA_D = DATA_D / "raw"
PROCESSED_D = DATA_D / "processed"


class DataLoader:
    """A class that acts like a list, but loads each item from storage on demand.

    Use as shown below. Note, output was shortened for clarity.

        >>> d = DataLoader(lower_case=True, include_newsgroup=True)
        >>> print(len(d))
        18846
        >>> print(d[824043])
        (['from', ... 'get', 'a', 'hold', 'of', 'him'], 'rec.sport.hockey')
        >>> d = DataLoader(lower_case=True, include_newsgroup=False)
        >>> print(d[8243])
        ['from', ... 'get', 'a', 'hold', 'of', 'him']
    """

    to_sep_pattern = re.compile("[^A-Za-z0-9]+")
    sep_pattern = re.compile("\s+")

    def __init__(self, raw_data_d=RAW_DATA_D, lower_case=True, include_newsgroup=True):
        """

        Arguments
        ---------
            `lower_case`: `bool`
                Whether to lower case everything.
            `include_newsgroup`: `bool`
                Whether to return the newsgroup that this example belongs to, or just return the tokens.

        Yields
        ------
            if `include_newsgroup=True`:
                `tuple` of list of tokens and the newsgroup. Like:
                    
                    (["I", "don", "t", "like", "politics"], "talk.politics.misc")
            else:
                A list of tokens

                    ["I", "don", "t", "like", "politics"]
        """

        self.all_files = list(raw_data_d.glob("*/*"))
        self.lower_case = lower_case
        self.include_newsgroup = include_newsgroup

    def _tokenize(self, content):
        if self.lower_case:
            content = content.lower()

        # Convert non alpha numeric symbols to space
        content = self.to_sep_pattern.sub(" ", content)
        # Separate tokens by space and remove empty tokens
        tokens = [i for i in self.sep_pattern.split(content) if i]
        return tokens

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
        file = self.all_files[idx]
        file = self.all_files[idx]
        newsgroup = file.parent.name
        with open(file, encoding="latin1") as f:
            content = f.read()

        tokens = self._tokenize(content)
        if self.include_newsgroup:
            return (tokens, newsgroup)
        return tokens

    def __len__(self):
        return len(self.all_files)


def numerize_data(raw_data_d=RAW_DATA_D, lower_case=True, min_count=20):
    """ Tokenizes our data and stores the token ids and newsgroup ids
        For easier processing later on.

        Arguments
        ---------
            `min_count`: `int`
                The minimum number of times a token must appear so that it is not
                replaced by "UNK", which stands for "Unknown"
    
       Outputs to file
       ---------------
           `data/processed/data.pkl`:
                A dictionary of the following format.

                    {
                        "id_to_tokens": id_to_tokens,
                        "id_to_newsgroups": id_to_newsgroups,
                        "newsgroup_and_token_ids_per_post": newsgroup_and_token_ids_per_post,
                    }
                Each value in dict is in the format that `create_term_newsgroup_matrix`
                expects(look at `vec_spaces.py`)
    """

    token_counts = {}
    newsgroups = set([])
    tokens_and_newsgroups = DataLoader(lower_case=lower_case, include_newsgroup=True)

    for tokens, doc in tokens_and_newsgroups:
        for tok in tokens:
            token_counts[tok] = token_counts.get(tok, 0) + 1
        newsgroups.add(doc)

    # Filter by min count
    token_counts = list(filter(lambda item: item[1] >= min_count, token_counts.items()))

    print("After filtering by min_count, {} tokens.".format(len(token_counts)))
    token_to_ids = {word: i for i, (word, _) in enumerate(token_counts)}
    id_to_tokens = {i: word for word, i in token_to_ids.items()}
    unknown_id = len(token_to_ids)
    token_to_ids["UNK"] = token_to_ids
    id_to_tokens[unknown_id] = "UNK"

    newsgroup_to_ids = {newsgroup: i for i, newsgroup in enumerate(newsgroups)}
    newsgroup_and_token_ids_per_post = [
        (translate(tokens, token_to_ids, unknown_id), newsgroup_to_ids[newsgroup])
        for tokens, newsgroup in tokens_and_newsgroups
    ]
    id_to_newsgroups = {i: newsgroup for newsgroup, i in newsgroup_to_ids.items()}

    numerized_data = {
        "id_to_tokens": id_to_tokens,
        "id_to_newsgroups": id_to_newsgroups,
        "newsgroup_and_token_ids_per_post": newsgroup_and_token_ids_per_post,
    }

    print(
        "A sample of the final numerized_data:\n"
        "\tid to tokens: {}...\n"
        "\tid to newsgroups: {}...\n"
        "\ttoken_ids_and_newsgroups:\n\t\t{}".format(
            _yield_few_times(iter(id_to_tokens.items()), 5),
            _yield_few_times(iter(id_to_newsgroups.items()), 5),
            "\n\t\t".join(
                "{}: {}...".format(y, x_token_ids[:5])
                for x_token_ids, y in newsgroup_and_token_ids_per_post[:5]
            ),
        )
    )

    PROCESSED_D.mkdir(exist_ok=True)
    with open(PROCESSED_D / "data.pkl", "wb") as fb:
        pkl.dump(numerized_data, fb)


def read_processed_data():
    """
    numerized_data = {
        "id_to_tokens": id_to_tokens,
        "id_to_newsgroups": id_to_newsgroups,
        "newsgroup_and_token_ids_per_post": newsgroup_and_token_ids_per_post,
    }
    """
    with open(PROCESSED_D / "data.pkl", "rb") as fb:
        numerized_data = pkl.load(fb)
    return numerized_data


def translate(x, dict_, unknown_becomes):
    return [dict_.get(key, unknown_becomes) for key in x]


def _yield_few_times(gen, times):
    return [next(gen) for i in range(times)]


def main():
    numerize_data()


if __name__ == "__main__":
    main()
