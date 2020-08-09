import numpy as np

def hit_rate(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list_k = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list_k)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate
    
    
def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)  # [False, False, True, True]
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision

    
def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    bought_list = np.array(bought_list)
    prices_recommended_k = np.array(prices_recommended[:k])
    recommended_list_k = np.array(recommended_list[:k])
    
    flags = np.isin(recommended_list_k, bought_list)

    revenue_of_rel_in_rec = np.dot(recommended_list_k[flags], 
                                 prices_recommended_k[flags])
    
    revenue_of_rec = np.dot(recommended_list_k, prices_recommended_k)
    
    money_precision = revenue_of_rel_in_rec / revenue_of_rec
    
    return money_precision

    
def mean_money_precision_at_k(recommended_array, bought_array, df_prices, k=5):
    """Функция для подсчета метрики money_precision по всем юзерам
    
    Input
    -----
    recommended_array: list, содержащий списки рекомендованных товаров
    bought_array: list, содержащий списки купленных товаров
    df_prices: pd.Dataframe c индексом 'item_id' и столбцом 'prices'
    
    Return
    ------
    mean_money_precision_at_k: float, среднее значение метрики по всем юзерам
    """
    
    assert len(recommended_array) == len(bought_array), 'Mismatched dimensions for recommended and bought series'
    
    recommended_array = np.array(recommended_array)
    bought_array = np.array(bought_array)
    
    recommended_prices_array = list(map((lambda x: get_prices(x, df_prices)), recommended_array))
    
    money_precision_list = [money_precision_at_k(recs, bought, prices, k=k)
                           for recs, bought, prices in zip(recommended_array, bought_array, recommended_prices_array)]
    
    mean_money_precision_at_k = np.mean(money_precision_list)
    
    return mean_money_precision_at_k
    

def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)  # [False, False, True, True]
    
    recall = flags.sum() / len(bought_list)
    
    return recall    


def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list_k = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list_k)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list_k = np.array(recommended_list[:k])
    prices_bought = np.array(prices_bought)
    
    flags = np.isin(bought_list, recommended_list_k)

    revenue_of_rel_in_rec = np.dot(bought_list[flags], prices_bought[flags])
    revenue_of_rel = np.dot(bought_list, prices_bought)
    
    recall = revenue_of_rel_in_rec / revenue_of_rel
    
    return recall
    
    
def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result


def reciprocal_rank(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    indexes = np.where(flags == True)
    if len(indexes) > 0 and len(indexes[0]) > 0:
        rank = indexes[0][0] + 1
    
    return (1 / rank) if rank else 0
    
def get_prices(recommended_list, df_prices):
    """Функция возвращает цены товаров для последующего использования в money_precision
    
    Input
    -----
    recommended_list: list или np.array с item_id рекомендуемых товаров
    df_prices: pd.Dataframe c индексом 'item_id' и столбцом 'prices'
    
    Return
    ------
    prices_recommended: list с ценами рекомендуемых товаров
    """
    assert 'item_id' == df_prices.index.name, "Incorrect input. Index shoud be named as 'item_id'"
    assert 'price' in df_prices.columns, "Incorrect input. Column 'price' is abscent"
    
    prices_recommended = []
    
    for item_id in recommended_list:
        price = df_prices.loc[df_prices.index == item_id, 'price'].values
        prices_recommended.extend(price)
        
    return prices_recommended