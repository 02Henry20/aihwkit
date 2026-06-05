import numpy as np

def get_coincidences(x_words, d_words):

    n = 0

    
    # Negative determined from selected sign bits
    negative = (((x_words[0] >> 31) & 1) ^ ((d_words[1] >> 31) & 1)) != 0

    # First word: exclude LSB/sign bit
    n += bin(x_words[0] & d_words[0] & 0xFFFFFFFE).count("1")

    # Remaining words: count all bits
    for i in range(1, len(x_words)):
        x = x_words[i]
        d = d_words[i]
        x_and_d = x & d

        n += bin(x_and_d).count("1")

    return n, negative



x_words1 = np.array([3965990158, 2426213978, 1891944214, 3564766615], dtype=np.uint32)
x_words2 = np.array([1078208544, 1644210177, 1216350376, 1073775112], dtype=np.uint32)

d_words1 = np.array([1403023646,1814040288, 17124356, 3961897093], dtype=np.uint32)
d_words2 = np.array([32768, 0 , 8 , 2701410304], dtype=np.uint32)
d_words3 = np.array([15204608, 469893120, 335773761, 859974701], dtype=np.uint32)
d_words4 = np.array([1346895912, 2216953600, 1210056704, 2294417488], dtype=np.uint32)


coincidences, negative = get_coincidences(x_words1, d_words1)
print(f"Coincidences: {coincidences}, Negative: {negative}")
coincidences, negative = get_coincidences(x_words1, d_words2)
print(f"Coincidences: {coincidences}, Negative: {negative}")
coincidences, negative = get_coincidences(x_words1, d_words3)
print(f"Coincidences: {coincidences}, Negative: {negative}")
# coincidences, negative = get_coincidences(x_words1, d_words4)
# print(f"Coincidences: {coincidences}, Negative: {negative}")
# coincidences, negative = get_coincidences(x_words2, d_words1)
# print(f"Coincidences: {coincidences}, Negative: {negative}")
# coincidences, negative = get_coincidences(x_words2, d_words2)
# print(f"Coincidences: {coincidences}, Negative: {negative}")
# coincidences, negative = get_coincidences(x_words2, d_words3)
# print(f"Coincidences: {coincidences}, Negative: {negative}")
coincidences, negative = get_coincidences(x_words2, d_words4)
print(f"Coincidences: {coincidences}, Negative: {negative}")
