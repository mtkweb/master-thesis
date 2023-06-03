from typing import List, Tuple

def calculate_levenshtein_distance(word_a: str, word_b: str) -> Tuple[int, List[int]]:
    if len(word_a) != len(word_b):
        raise ValueError('Words must be of the same length')

    # Actually, we now calculate the Hamming distance, but that's fine for our purposes
    number_of_different_letters = 0
    different_positions = []
    for i in range(len(word_a)):
        if word_a[i] != word_b[i]:
            number_of_different_letters += 1
            different_positions.append(i)

    return number_of_different_letters, different_positions

def find_minimal_pairs(words: List[str]) -> List[Tuple[Tuple[str, str], int]]:
    minimal_pairs: List[Tuple[Tuple[str, str], int]] = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            levenshtein_distance, different_positions = calculate_levenshtein_distance(words[i], words[j])
            if levenshtein_distance == 1:
                minimal_pairs.append(((words[i], words[j]), different_positions[0]))
    return minimal_pairs

