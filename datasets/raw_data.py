#%%
import pandas as pd
#%%
def load_raw_data(dataset):
    if dataset == "abalone":
        data = pd.read_csv('./data/abalone.data', header=None)
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Sex")
        columns.remove("Rings")
        continuous_features = columns
        categorical_features = [
            "Sex",
            "Rings"
        ]
        integer_features = []
        ClfTarget = "Rings"
    
    elif dataset == "banknote":
        data = pd.read_csv('./data/data_banknote_authentication.txt', header=None)
        data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            "variance", "skewness", "curtosis", "entropy"
        ]
        categorical_features = [
            'class',
        ]
        integer_features = []
        ClfTarget = "class"
        
    elif dataset == "breast":
        data = pd.read_csv('./data/wdbc.data', header=None)
        data = data.drop(columns=[0]) # drop ID number
        columns = ["Diagnosis"]
        common_cols = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave points",
            "symmetry",
            "fractal dimension",
        ]
        columns += [f"{x}1" for x in common_cols]
        columns += [f"{x}2" for x in common_cols]
        columns += [f"{x}3" for x in common_cols]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = []
        continuous_features += [f"{x}1" for x in common_cols]
        continuous_features += [f"{x}2" for x in common_cols]
        continuous_features += [f"{x}3" for x in common_cols]
        categorical_features = [
            "Diagnosis"
        ]
        integer_features = []
        ClfTarget = "Diagnosis"
        
    elif dataset == "concrete":
        data = pd.read_csv('./data/Concrete_Data.csv')
        columns = [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate",
            "Age",
            "Concrete compressive strength"
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Age")
        continuous_features = columns
        categorical_features = [
            "Age",
        ]
        integer_features = []
        ClfTarget = "Age"
        
    elif dataset == "kings":
        data = pd.read_csv('./data/kc_house_data.csv')
        
        continuous_features = [
            'price', 
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'lat',
            'long',
            'sqft_living15',
            'sqft_lot15',
        ]
        categorical_features = [
            'bedrooms',
            'bathrooms',
            'floors',
            'waterfront',
            'view',
            'condition',
            'grade', 
        ]
        integer_features = [
            'price',
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'sqft_living15',
            'sqft_lot15',
        ]
        ClfTarget = "grade"
        
    elif dataset == "letter":
        data = pd.read_csv('./data/letter-recognition.data', header=None)
        columns = [
            "lettr",
            "x-box",
            "y-box",
            "width",
            "high",
            "onpix",
            "x-bar",
            "y-bar",
            "x2bar",
            "y2bar",
            "xybar",
            "x2ybr",
            "xy2br",
            "x-ege",
            "xegvy",
            "y-ege",
            "yegvx",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("lettr")
        continuous_features = columns
        categorical_features = [
            "lettr"
        ]
        integer_features = columns
        ClfTarget = "lettr"
        
    elif dataset == "loan":
        data = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        
        continuous_features = [
            'Age',
            'Experience',
            'Income', 
            'CCAvg',
            'Mortgage',
        ]
        categorical_features = [
            'Family',
            'Personal Loan',
            'Securities Account',
            'CD Account',
            'Online',
            'CreditCard'
        ]
        integer_features = [
            'Age',
            'Experience',
            'Income', 
            'Mortgage'
        ]
        data = data[continuous_features + categorical_features]
        data = data.dropna()
        ClfTarget = "Personal Loan"
        
    elif dataset == "redwine":
        data = pd.read_csv('./data/winequality-red.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif dataset == "spam":
        data = pd.read_csv('./data/spambase.data', header=None)
        columns = [
            "word_freq_make",
            "word_freq_address",
            "word_freq_all",
            "word_freq_3d",
            "word_freq_our",
            "word_freq_over",
            "word_freq_remove",
            "word_freq_internet",
            "word_freq_order",
            "word_freq_mail",
            "word_freq_receive",
            "word_freq_will",
            'word_freq_people',
            "word_freq_report",
            'word_freq_addresses',
            "word_freq_free",
            "word_freq_business",
            "word_freq_email",
            "word_freq_you",
            'word_freq_credit',
            'word_freq_your',
            "word_freq_font",
            'word_freq_000',
            'word_freq_money',
            "word_freq_hp",
            'word_freq_hpl',
            'word_freq_george',
            "word_freq_650",
            "word_freq_lab",
            "word_freq_labs",
            'word_freq_telnet',
            "word_freq_857",
            "word_freq_data",
            'word_freq_415',
            "word_freq_85",
            "word_freq_technology",
            "word_freq_1999",
            "word_freq_parts",
            'word_freq_pm',
            "word_freq_direct",
            "word_freq_cs",
            "word_freq_meeting",
            'word_freq_original',
            'word_freq_project',
            'word_freq_re',
            'word_freq_edu',
            'word_freq_table',
            'word_freq_conference',
            "char_freq_;",
            "char_freq_(",
            "char_freq_[",
            "char_freq_!",
            "char_freq_$",
            "char_freq_#",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
            "class"
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = [
            "class"
        ]
        integer_features = [
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
        ]
        ClfTarget = "class"
        
    elif dataset == "whitewine":
        data = pd.read_csv('./data/winequality-white.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif dataset == "yeast":
        data = pd.read_csv('./data/yeast.data', delimiter="  ", index_col=False, header=None)
        columns = [
            "Sequence Name",
            "mcg",
            "gvh",
            "alm",
            "mit",
            "erl",
            "pox",
            "vac",
            "nuc",
            "localization_site"
        ]
        data.columns = columns
        data = data.drop(columns=["Sequence Name"])
        
        data = data.dropna() # remove 5 rows
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            "mcg",
            "gvh",
            "alm",
            "mit",
            "erl",
            "pox",
            "vac",
            "nuc",
        ]
        categorical_features = [
            "localization_site"
        ]
        integer_features = []
        ClfTarget = "localization_site"

    elif dataset == "covtype":
        data = pd.read_csv('./data/covtype.csv')
        # data = data.sample(frac=1, random_state=0).reset_index(drop=True)
        # data = data.dropna(axis=0)
        # data = data.iloc[:50000]
        
        continuous_features = [
            'Elevation',
            'Aspect', 
            'Slope',
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points',
        ]
        categorical_features = [
            'Cover_Type',
        ]
        integer_features = continuous_features
        ClfTarget = "Cover_Type"
    
    return data, continuous_features, categorical_features, integer_features, ClfTarget