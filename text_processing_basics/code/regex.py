import re
from pathlib import Path

RAW_DUMP_XML = Path("raw_data/Wikipedia.xml")


def count_regexp():
    """Counts the occurences of the regular expressions you will write.
    """
    # Here's an example regular expression that roughly matches a valid email address.
    # The ones you write below should be shorter than this
    email = re.compile("[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[a-zA-Z]{2,5}")

    ###### Write below #########
    subheading = re.compile("\=\=+.*\=\=+")
    link_to_subheading = re.compile("\[\[[\w\'*\-*\:*\(*\)*\_*\s*]+[#][\s*\w\\'*\-*\:*\(*\)*\_*s*]+\|*")
    doi_citation = re.compile("\{\{[c][ite](?!{{).*[dD][oO][iI]\s*[:|,=\/]*\s*[0-9]+\.[0-9]+.*\}\}")
    ###### End of your work #########

    patterns = {
        "emails": email,
        "subheadings": subheading,
        "links to subheadings": link_to_subheading,
        "citations with DOI numbers": doi_citation,
    }

    with open(RAW_DUMP_XML, encoding="utf-8") as f:
        dump_text = f.read()
        for name, pattern in patterns.items():
            if pattern is None:
                continue
            matches = pattern.findall(dump_text)
            count = len(matches)

            example_matches = [matches[i * (count // 5)] for i in range(5)]

            print("Found {} occurences of {}".format(count, name))
            print("Here are examples:")
            print("\n".join(example_matches))
            print("\n")


if __name__ == "__main__":
    count_regexp()
