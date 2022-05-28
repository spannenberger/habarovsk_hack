import pandas as pd
df = pd.read_csv("via_export_csv (2).csv", sep=",")
n_df = df["region_shape_attributes"].apply(lambda x: eval(x))
new_df = n_df.apply(pd.Series)
new_df = pd.concat([df["filename"], new_df], axis=1)
new_df = new_df[["filename", "x", "y", "width", "height"]]
new_df = new_df.rename(columns={"filename": "file_name", "x": "x_from", "y": "y_from"})
new_df["class"] = "walrus"
new_df.to_csv("train_data_new_2.csv", index=False)
print(new_df)
