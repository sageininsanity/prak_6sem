HEADER = ["SEX","INSR_BEGIN","INSR_END","EFFECTIVE_YR","INSR_TYPE","INSURED_VALUE","PREMIUM","OBJECT_ID","PROD_YEAR","SEATS_NUM","CARRYING_CAPACITY","TYPE_VEHICLE","CCM_TON","MAKE","USAGE","CLAIM_PAID"]

NUM_FEATURES = ["INSURED_VALUE", "PREMIUM", "SEATS_NUM", "CARRYING_CAPACITY", "PROD_YEAR", "EFFECTIVE_YR", "INSR_BEGIN", "INSR_END", "CLAIM_PAID"]

CAT_FEATURES = ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "USAGE"]

IMPROVED_HEADER = ['SEX_0.0', 'SEX_1.0', 'SEX_2.0', 'SEX_nan', 'INSR_TYPE_1201.0',
       'INSR_TYPE_1202.0', 'INSR_TYPE_1204.0', 'INSR_TYPE_nan',
       'TYPE_VEHICLE_Automobile', 'TYPE_VEHICLE_Bus',
       'TYPE_VEHICLE_Motor-cycle', 'TYPE_VEHICLE_Pick-up',
       'TYPE_VEHICLE_Special construction', 'TYPE_VEHICLE_Station Wagones',
       'TYPE_VEHICLE_Tanker', 'TYPE_VEHICLE_Tractor',
       'TYPE_VEHICLE_Trade plates', 'TYPE_VEHICLE_Trailers and semitrailers',
       'TYPE_VEHICLE_Truck', 'TYPE_VEHICLE_nan', 'USAGE_Agricultural Any Farm',
       'USAGE_Agricultural Own Farm', 'USAGE_Ambulance', 'USAGE_Car Hires',
       'USAGE_Fare Paying Passengers', 'USAGE_Fire fighting',
       'USAGE_General Cartage', 'USAGE_Learnes', 'USAGE_Others',
       'USAGE_Own Goods', 'USAGE_Own service', 'USAGE_Private',
       'USAGE_Special Construction', 'USAGE_Taxi', 'USAGE_nan']