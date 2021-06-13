import random
import pandas as pd
from time import time, sleep


def log(message):
    print(f"[+] {message}\n")

    
def generate_random_score():
    return random.randint(60, 100)


def generate_list_of_random_scores(total_records=1000):
    return list(map(lambda x: generate_random_score(), range(total_records)))


def generate_event_time_records(num_records):
    time_value = int(round(time()))
    output = pd.Series([time_value]*num_records, 
                       dtype="float64")
    
    return output


def main():
    log("Generating column values for math, science, technology")
    math = generate_list_of_random_scores()
    science = generate_list_of_random_scores()
    technology = generate_list_of_random_scores()
    
    log("Generating column values for random1 and random2")
    random1 = generate_list_of_random_scores()
    random2 = generate_list_of_random_scores()
    sex = ["male"] * 800 + ["female"] * 200

    all_df = pd.DataFrame(dict(sex=sex, 
                               math=math,
                               science=science,
                               technology=technology,
                               random1=random1,
                               random2=random2))

    log("Computing values for the approved column")
    all_df['approved'] = all_df.apply(lambda row: (row.math + row.science + row.technology) > 240, axis=1)
    approved_col = all_df.pop("approved")
    all_df.insert(0, "approved", approved_col)
    all_df.loc[0:599, 'approved'] = True
    
    log("Generating the index and event_time column values")
    all_df['index'] = range(1, len(all_df) + 1)
    all_df['event_time'] = generate_event_time_records(len(all_df))
    
    print(all_df)

    log("Converting approved and sex columns")
    all_df['approved'] = all_df.apply(lambda row: 1 if row.approved else 0, axis=1)
    all_df['sex'] = all_df.apply(lambda row: 1 if row.sex == "male" else 0, axis=1)

    print(all_df)
    
    return all_df


log("Running the main() function")
all_df = main()