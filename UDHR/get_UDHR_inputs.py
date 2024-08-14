def get_rights_lists():
    fn = "UDHR/UDHR_raw_rights.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    q_current, q_ideal = [], []
    for l in lines:
        inps = l.strip().split("{people}")
        q_current.append("{people} " + inps[1].strip())
        q_ideal.append("{people} " + inps[2].strip())
    return q_current, q_ideal


def get_identities_dicts():
    fn = "UDHR/UDHR_raw_identities.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    i_current = {}
    for l in lines:
        if l.startswith("ind: "):
            i_key = l.split("ind: ")[-1].split("(")[0].strip()
            i_current[i_key] = []
        else:
            _l = [_l.strip() for _l in l.split(",") if len(_l) > 1]
            i_current[i_key] += _l
    return i_current


if __name__ == "__main__":
    get_rights_lists()
    get_identities_dicts()
