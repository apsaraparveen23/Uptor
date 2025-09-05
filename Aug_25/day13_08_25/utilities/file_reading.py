import  pandas as pd

"""below is the user defined function for converting dat ain to dataframe"""
def data_to_data_frame(hello):
    """below code is with exception handling for data frame conversion"""
    try:
        df=pd.DataFrame(hello)
        return df
    except Exception as ex:
        return ex
