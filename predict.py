import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import sqlite3


class Predictor:
    def __init__(self):
        self.model = self.load_model()
        self.scale = self.scale_model()
        #self._setup_database()


    #def _setup_database(self):
        #conn = sqlite3.connect("profits_db.db")
        #cursor = conn.cursor()
        #sql_command = """CREATE TABLE IF NOT EXISTS profits_predictions (
                                                #id integer PRIMARY KEY,
                                                #profit real NOT NULL); """

        #cursor.execute(sql_command)
        #conn.commit()
        #conn.close()


    def _write_prediciton_to_database(self, pred):
        conn = sqlite3.connect("profits_db.db")
        cursor = conn.cursor()
        sql_command = """CREATE TABLE IF NOT EXISTS profits_predictions (
                                                        id integer PRIMARY KEY,
                                                        profit real NOT NULL); """

        cursor.execute(sql_command)
        conn.commit()
        cursor.execute("insert into profits_predictions values (NULL, ?)", (float(pred),))
        #never forget this, if you want the changes to be saved:
        conn.commit()
        conn.close()

    @staticmethod
    def load_model():
        """
        code to load model from the disk with pickle
        """
        model = pickle.load(open('finalized_model.pkl', 'rb'))
        return model

    @staticmethod
    def scale_model():
        scale = pickle.load(open("scaling.pkl", "rb"))
        return scale


    def predict_for_new_sample(self, json_data):
        #json_file = open(file)
        #json_str = json_file.read()
        #json_data = json.loads(json_str)
        l = ['Sales', 'Quantity', 'Discount', "Sub-Category", "Order Date"]
        print(json_data)
        for dic in json_data:
            out = {x: dic[x] for x in l}
        out['Order Quarter_' + str(int(out['Order Date'][:2]) // 4 + 1)] = int(out['Order Date'][:2]) // 4 + 1

        out["Discounts"] = float(out["Sales"]) * float(out["Discount"])
        del out["Discount"], out["Order Date"]
        out['Sales'] = float(out['Sales'])
        out['Quantity'] = float(out['Quantity'])
        out['Discounts'] = float(out['Discounts'])
        feature_vector = [out['Sales'], out['Quantity'], out['Discounts']]
        to_scale = self.scale.transform([np.array(feature_vector)])
        new1_vector = to_scale.tolist()
        new_vector = new1_vector[0]

        category_list = ['Category_Accessories',
                         'Category_Appliances', 'Category_Art', 'Category_Binders',
                         'Category_Bookcases', 'Category_Chairs', 'Category_Copiers',
                         'Category_Envelopes', 'Category_Fasteners', 'Category_Furnishings',
                         'Category_Labels', 'Category_Machines', 'Category_Paper',
                         'Category_Phones', 'Category_Storage', 'Category_Supplies',
                         'Category_Tables', 'Order Quarter_1', 'Order Quarter_2',
                         'Order Quarter_3', 'Order Quarter_4']
        for category in category_list:
            start_len = len(new_vector)
            for val in out.values():
                if type(val) == str and val in category:
                    new_vector.append(1)
                    break

            if len(new_vector) == start_len:
                new_vector.append(0)

        sample = [np.array(new_vector)]
        result = self.model.predict(sample)[0][0]
        print(f"the result is {result}")
        self._write_prediciton_to_database(result)
        return result

if __name__ == "__main__":
    with open("NewLineProfit.json", 'r') as json_file:
        sample = json.load(json_file)
    predictor = Predictor()
    my_result = predictor.predict_for_new_sample(sample)
    print(f"the final result is {my_result}")
    #print(my_result[0][0])