# Advances in Data Mining - Assignment 2: Stream Mining

My solutions for the second programming assignment of the Advances in Data Mining course. The assignment covers three classic stream mining algorithms: reservoir sampling, Bloom filter and Flajolet-Martin probabilistic counting. Full assignment text is in **Assignment_2__Stream_Mining.pdf** and the original skeleton files are in the zip next to it.

Author: Ali Momen (s4993128)

---

## Contents

| File | Task | What I implemented |
|---|---|---|
| **task1_4993128.py** | 1. Reservoir sampling | body of **reservoir_sampling** |
| **task2_4993128.py** | 2. Bloom filter | **create_hash_functions**, **add_to_bloom_filter**, **check_bloom_filter** |
| **task3_4993128.py** | 3. Flajolet-Martin | **count_trailing_zeros** and **add** |
| **validate_submission.py** | - | checker from the course, untouched |

Only the code between **# BEGIN IMPLEMENTATION** and **# END IMPLEMENTATION** is mine. The rest of the skeleton is kept as is, like the assignment asks.

---

## Setup and running

I ran everything with Python 3.14.5 on my machine (Windows 11), global interpreter, no virtual environment. The task files need only **numpy** (task 2) and the standard library; the validator needs **pytest**.

```
pip install numpy pytest
```

Note that **requirements.txt** is the general one from my course template, so it lists a lot more (e.g. torch, gymnasium) than this assignment really uses.

Each task file has a main block at the bottom for quick testing. Run them from repo root:

```
python task1_4993128.py
python task2_4993128.py
python task3_4993128.py
python -m pytest validate_submission.py
```

Task 1 prints the whole sample of 5000 numbers, so the output is long. Task 2 finishes in under a second and task 3 in about 2 seconds. The validator passes (3 passed).

---

## Task 1 - Reservoir sampling

Goal: keep a uniform random sample of size $k$ from a stream of unknown length $N$, without storing the stream.

How **reservoir_sampling** does it:

- first $k$ items go straight into the reservoir
- for the item at (0-based) position $i \ge k$, draw $j$ uniformly from $\{0, \dots, i\}$ with **random.randint(0, index)**
- if $j < k$ the item replaces **sample[j]**, otherwise it is dropped

So item $i$ gets in with probability $k/(i+1)$, and every later item $t$ throws it out with probability $1/(t+1)$. Put together:

$$P(\text{item } i \text{ in final sample}) = \frac{k}{i+1} \prod_{t=i+1}^{N-1} \frac{t}{t+1} = \frac{k}{N}$$

And that is the same for every item, which is exactly what "representative" means here. Memory stays at $k$ items the whole time. The main block samples $k = 5000$ out of 10,000 Gaussian values.

---

## Task 2 - Bloom filter

Notation: $n$ is the number of stored keys, $m$ the size of the bit array and $k$ the number of hash functions.

The hash function with index $i$ is SHA-256 of the string **f"{i}{x}"**, turned into an integer and taken mod $m$. Each lambda gets **i=i** as a default argument; without it all lambdas would share the last value of **i** because of late binding in Python, so all $k$ functions would be the same one.

- **add_to_bloom_filter** sets the $k$ bits of the account to 1
- **check_bloom_filter** returns False as soon as one of the $k$ bits is 0, otherwise True

A stored key always finds all its bits set, so there are no false negatives and the TPR (true positive rate) is always 1. For the FPR (false positive rate) the formula from the slides is

$$\text{FPR}(k) \approx \left(1 - e^{-kn/m}\right)^k, \qquad k_{opt} = \frac{m}{n} \ln 2$$

### Hash function sweep

Same setup as in the main block: $n$ = 100,000 real accounts, $m = 8n$ bits and 100,000 fake accounts as negatives. I looped $k$ from 2 to 30 (selected rows below):

| $k$ | measured FPR | formula |
|---|---|---|
| 2 | 0.0491 | 0.0489 |
| 3 | 0.0308 | 0.0306 |
| 4 | 0.0245 | 0.0240 |
| 5 | 0.0219 | 0.0217 |
| **6** | **0.0207** | 0.0216 |
| 7 | 0.0218 | 0.0229 |
| 8 | 0.0244 | 0.0255 |
| 10 | 0.0333 | 0.0342 |
| 15 | 0.0823 | 0.0823 |
| 20 | 0.1795 | 0.1803 |
| 30 | 0.4913 | 0.4897 |

FPR goes down until $k = 6$ and then climbs again; at $k = 30$ almost half of the fake accounts get through. The formula gives $k_{opt} = 8 \ln 2 \approx 5.55$, so 5 or 6 hash functions, and the measured minimum is at 6. So the experiment agrees with the slides. TPR was 1.0 for every $k$.

The main block still uses 2 hash functions like in the skeleton. With that it prints FPR 0.049 and 22.1% of bits set, which is what $1 - e^{-2n/8n} = 0.221$ predicts.

---

## Task 3 - Flajolet-Martin

Notation: $n$ is the number of hash functions (**num_hashes**) and $R_i$ the maximum number of trailing zeros seen so far for hash function $i$.

- **count_trailing_zeros** converts the number to binary with **bin(x)[2:]**, walks from the right end and counts zeros until the first 1
- **add** hashes the item once per seed $i$ (SHA-256 of **"{seed}-{item}"**), counts the trailing zeros and keeps the maximum in **max_trailing_zeros[i]**

**estimate_number** comes from the skeleton. It takes the median of the $R_i$ and returns

$$\hat{D} = \frac{2^{\,\text{median}(R_1, \dots, R_n)}}{0.75351}$$

where $\hat{D}$ is the estimated number of distinct items.

### Choosing n

Results on 100,000 distinct items:

| $n$ | median $R$ | estimate |
|---|---|---|
| 1 | 19 | 695,794 |
| 5 | 16 | 86,974 |
| 10 | 16 | 86,974 |
| 20 | 16 | 86,974 |
| 40 | 16 | 86,974 |
| 80 | 16 | 86,974 |

With one hash function the estimate is 7 times too big, since one lucky hash value with many trailing zeros decides everything. From 5 on the median stays at 16 and the estimate is about 13% under the real count. That is as close as this estimator gets here: it can only output a power of two divided by a constant (43,487, 86,974, 173,948, ...) and 86,974 is the nearest of those to 100,000. The main block uses $n = 20$, which is plenty.

One small thing: the constant in the original Flajolet-Martin paper is $\varphi \approx 0.77351$, the skeleton has 0.75351. It is outside the implementation area so I left it alone; with 0.77351 the estimate would be 84,725.

---

## Repository layout

```
.
├── task1_4993128.py                  reservoir sampling
├── task2_4993128.py                  Bloom filter
├── task3_4993128.py                  Flajolet-Martin
├── validate_submission.py            submission checker from the course
├── Assignment_2__Stream_Mining.pdf   assignment text
├── Programming Ass. 2 Stream Mining attached files 26 September 2026 1014.zip
│                                     original skeleton, PDF and checker
├── requirements.txt
└── LICENSE
```

---

## License

MIT, see **LICENSE**.
