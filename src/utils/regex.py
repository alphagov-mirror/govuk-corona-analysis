def coronavirus_misspellings_and_typos_regex():
    """Return a regex you can use to replace misspellings of coronavirus with the correct spelling
    >>> import re
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'caronavirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coranavirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'cornavirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'cornovirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronaviris')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'cornoavirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronavirius')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronavirous')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronaviru')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronaviurs')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronavius')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronoavirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronovirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coronvirus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'corona iris')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'corona virus')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'covid')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'corona')]
    ['coronavirus']
    >>> [re.sub(coronavirus_misspellings_and_typos_regex(), 'coronavirus', 'coron')]
    ['coronavirus']
    """
    return "coronavirus|covid19|covid.19|caronavirus|coranavirus|cornavirus|cornovirus|coronaviris|cornoavirus|" + \
           "coronavirius|coronavirous|coronaviru|coronaviurs|coronavius|coronoavirus|coronovirus|coronvirus|" + \
           "corona iris|corona virus|covid|corona|coron"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
