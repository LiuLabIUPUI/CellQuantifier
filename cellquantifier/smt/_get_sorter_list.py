def get_sorter_list(phys_df):
    sorter_list = []
    for column_name in phys_df.columns:
        if "sort_flag" in column_name:
            sorter_list.append(column_name)

    return sorter_list
