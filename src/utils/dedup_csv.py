# >>> import pandas as pd
# >>> toclean = pd.read_csv('1606251790_0B3CC61FD4764094EAEA375353022009_G3486172.csv')
# >>> len(toclean)
# 1891157
# >>> deduped = toclean.drop_duplicates('RP_STORY_ID')
# >>> len(deduped)
# 701071
# >>> deduped = deduped.drop_duplicates('EVENT_TEXT')
# >>> len(deduped)
# 320946
# >>> deduped.to_csv('1606251790_0B3CC61FD4764094EAEA375353022009_G3486172_deduped.csv', index=False)
import pandas as pd

FILE_NAME = '1606251790_0B3CC61FD4764094EAEA375353022009_G3486172'

toclean = pd.read_csv(f'{FILE_NAME}.csv')
deduped = toclean.drop_duplicates('RP_STORY_ID')
deduped = deduped.drop_duplicates('EVENT_TEXT')
deduped.to_csv(f'{FILE_NAME}_deduped.csv', index=False)
