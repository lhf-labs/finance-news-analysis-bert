import pandas as pd

if __name__ == '__main__':
    FILE_NAME = '../data/2020-08-11to2021-02-11'

    toclean = pd.read_csv(f'{FILE_NAME}.csv')
    deduped = toclean.drop_duplicates('RP_STORY_ID')
    deduped = deduped.drop_duplicates('EVENT_TEXT')
    deduped = deduped.drop_duplicates('HEADLINE')
    deduped = deduped[~deduped['EVENT_SENTIMENT_SCORE'].isna()]
    deduped = deduped[pd.to_numeric(deduped['EVENT_SENTIMENT_SCORE'], errors='coerce').notnull()]
    deduped.to_csv(f'{FILE_NAME}_deduped.csv', index=False)
