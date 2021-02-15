import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


scaling = pickle.load(open("scaling.pkl", "rb"))

category_list = ['Category_Accessories',
                'Category_Appliances', 'Category_Art', 'Category_Binders',
                'Category_Bookcases', 'Category_Chairs', 'Category_Copiers',
                'Category_Envelopes', 'Category_Fasteners', 'Category_Furnishings',
                'Category_Labels', 'Category_Machines', 'Category_Paper',
                'Category_Phones', 'Category_Storage', 'Category_Supplies',
                'Category_Tables', 'Order Quarter_1', 'Order Quarter_2',
                'Order Quarter_3', 'Order Quarter_4']


def get_right_features(file):
    json_file = open(file)
    json_str = json_file.read()
    json_data = json.loads(json_str)
    l = ['Sales', 'Quantity', 'Discount', "Sub-Category", "Order Date"]
    for dic in json_data:
        out = {x: dic[x] for x in l}
    out['Order Quarter_' + str(int(out['Order Date'][:2]) // 4 + 1)] = int(out['Order Date'][:2]) // 4 + 1

    out["Discounts"] = float(out["Sales"]) * float(out["Discount"])
    del out["Discount"], out["Order Date"]
    out['Sales'] = float(out['Sales'])
    out['Quantity'] = float(out['Quantity'])
    out['Discounts'] = float(out['Discounts'])
    feature_vector = [out['Sales'], out['Quantity'], out['Discounts']]
    scaled = scaling.transform([np.array(feature_vector)])[0]
    new_vector = scaled.tolist()
    print(f"new vector after scaling is {new_vector}")

    print("output is: " + str(out))
    print(out.values())
    for category in category_list:
        start_len = len(new_vector)
        for val in out.values():
            if type(val) == str and val in category:
                new_vector.append(1)
                break

        if len(new_vector) == start_len:
            new_vector.append(0)


    print(f"new vector is {new_vector}")
    print(np.array(new_vector))
    print(len(new_vector))

######################################################################

get_right_features("NewLineProfit.json")



