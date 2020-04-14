import pandas as pd
from openpyxl.workbook import Workbook
import re

data_uiszendesk = pd.read_excel(io = '../../data/GAUISZ_Draft_Report_20200401to20200407.xlsx', 
                                sheet_name = 'Data tab', 
                                col_index = None)

# remove new lines
#data_uiszendesk['Q3_or_Description']replace('\n',' ', regex=True)
# remove everything 'referrer'-related
data_uiszendesk['Q3_or_Description'] = data_uiszendesk['Q3_or_Description'].str.split('\[Referrer\]|referrer\:').str[0]
# remove everything 'user_agent'-related
data_uiszendesk['Q3_or_Description'] = data_uiszendesk['Q3_or_Description'].str.split('\[User agent\]|user_agent\:').str[0]
# remove everything 'javascript_enable'-related
data_uiszendesk['Q3_or_Description'] = data_uiszendesk['Q3_or_Description'].str.split('\[JavaScript Enabled\]|javascript_enabled\:').str[0]

data_uiszendesk.to_excel("../../data/test.xlsx")





