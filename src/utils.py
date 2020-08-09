import pandas as pd
import numpy as np

def prefilter_items(data, item_features=None, take_n_popular=5000, 
                    filters={'top_popular': 0.5, 
                             'top_notpopular': 0.001,
                             'cheaper_than': 1,
                             'more_expensive_than': 20,
                             'last_purchase_n_weeks_back': 20,
                             'department_filter': 0.04
                            }):
    """Предфильтрация товаров
    
    Input
    ------
    data: pd.DataFrame, 
          Содержит столбцы 'item_id', 'user_id', 'sales_value', 'quantity', 'week_no'
          
    item_features: pd.DataFrame,
                   Содержит фичи товаров. Необходим для фильтрации по категориям. 
    
    take_n_popular: int 
                    Количество возвращаемых товаров
    filters: dict
             словарь, в котором ключами являются названия фильтров, которые необходимо применить, а значениями пороги, 
             по которым необходимо фильтровать. По умолчанию применяются все фильтры, кроме 'department_filter'
             Возможные значения: 'top_popular', 'top_notpopular', 'cheaper_than', 'more_expensive_than',
                               'last_purchase_n_weeks_back', 'department_filter'
    
    Return
    ------
    data - dataframe, отфильтрованнный в соответствии с указанными filters и содержащий топ-take_n_popular товаров
    """
    # Проверяем, что фильтры указаны верно
    default_filters = set(['top_popular', 'top_notpopular', 'cheaper_than', 'more_expensive_than', 
                          'last_purchase_n_weeks_back', 'department_filter'])
    
    assert all(filter_ in default_filters for filter_ in filters.keys()), ('Invalid filters\' keys. '
                                                                           'Please check help for '
                                                                           'getting valid values')
    
    filtered_items = set()
    
    # 0. Транзакции с нулевым количеством и стоимостью
    train_idx_todrop = data.loc[(data['quantity'] == 0) & (data['sales_value'] <= 0.1)].index
    data = data.drop(index=train_idx_todrop, axis=0).reset_index(drop=True)
    data.loc[data['quantity'] == 0, 'quantity'] = 1
    
    # 1. Неинтересные категории
    if filters.get('department_filter'):
        assert item_features is not None, 'Function expects \'item_features\' dataframe for department filter'
        
        # Фильтруем по категориям
        train_categories = pd.merge(data[['item_id', 'user_id']], \
                                    item_features[['item_id', 'sub_commodity_desc']], how='left')
        nunique_users = train_categories['user_id'].nunique()

        categories_popularity = train_categories.groupby('sub_commodity_desc')['user_id'].nunique().reset_index()
        categories_popularity['user_id'] = categories_popularity['user_id'] / nunique_users
        categories_popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
        
        not_popular_categories = categories_popularity.loc[\
                                    categories_popularity['share_unique_users'] < filters.get('department_filter'),
                                                           'sub_commodity_desc'
                                                          ].tolist()
        
        unique_items_in_npc = train_categories.loc[train_categories['sub_commodity_desc'].\
                                                   isin(not_popular_categories), 'item_id'].unique()
        
        filtered_items.update(unique_items_in_npc)
        
    
    # Фильтрация из урока
    # Подготовка для фильтрации по популярности
    if filters.get('top_popular') is not None or filters.get('top_notpopular') is not None:
        items_popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
        items_popularity['user_id'] = items_popularity['user_id'] / data['user_id'].nunique()
        items_popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
        items_popularity = items_popularity.sort_values(by=['share_unique_users']).reset_index(drop=True)
        
    # Самые популярные (итак купят)
    if filters.get('top_popular') is not None:
        top_popular = items_popularity[items_popularity['share_unique_users'] >= filters.get('top_popular')]\
                                        .item_id.tolist()
        filtered_items.update(top_popular)
    
    # Самые непопулярные (итак не купят)
    if filters.get('top_notpopular') is not None:
        top_notpopular = items_popularity[items_popularity['share_unique_users'] <= filters.get('top_notpopular')]\
                                        .item_id.tolist()
        filtered_items.update(top_notpopular)
    
    # Подготовка для фильтрации по цене
    if filters.get('cheaper_than') is not None or filters.get('more_expensive_than') is not None:
        df_prices = data.groupby('item_id')[['quantity', 'sales_value']].sum()
        df_prices['price'] = df_prices.apply(lambda item: 
                                             item['sales_value'] / item['quantity']
                                             if item['quantity'] else 0, axis=1
                                            )
        df_prices.drop(columns=['quantity', 'sales_value'], inplace=True)
        df_prices.reset_index(inplace=True)
        
    # 2. Удаление товаров, со средней ценой < 1$
    if filters.get('cheaper_than') is not None:
        low_price = df_prices.loc[df_prices['price'] < filters.get('cheaper_than'), 'item_id'].unique().tolist()
        filtered_items.update(low_price)
    
    # 3. Удаление товаров со соедней ценой > 20$
    if filters.get('more_expensive_than') is not None:
        high_price = df_prices.loc[df_prices['price'] > filters.get('more_expensive_than'), 
                                   'item_id'].unique().tolist()
        filtered_items.update(high_price)
    
    # 4. Товары, которые не продавались 20 недель
    if filters.get('last_purchase_n_weeks_back') is not None:
        n_weeks_back = data['week_no'].unique().max() - filters.get('last_purchase_n_weeks_back')
        item_ids_before = set(data[data['week_no'] <= n_weeks_back].item_id.tolist())
        item_ids_after = set(data[data['week_no'] > n_weeks_back].item_id.tolist())
        item_ids_last_purchase_n_weeks_back = list(item_ids_before.difference(item_ids_after))
        filtered_items.update(item_ids_last_purchase_n_weeks_back)
        
    data = data[~data['item_id'].isin(filtered_items)]
    
    data = pd.merge(data, item_features[['item_id', 'sub_commodity_desc']], how='left')
    
    # 5. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_n = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    data.loc[~data['item_id'].isin(top_n), 'item_id'] = 999999
    
    return data
    
def create_prices_df(train_data):
    """Функция для формирования датасета с ценами товаров
    Input
    -----
    train_data: pd.Dataframe со столбцами 'sales_value' и 'quantity'
    
    Return
    ------
    df_prices: pd.Dataframe c индексом 'item_id' и столбцом 'prices' 
    """
    assert all(col in train_data.columns for col in ['quantity', 'sales_value']),\
                    "Incorrect input. Columns 'quantity' и 'sales_value' are abscent"
    
    df_prices = train_data.groupby('item_id')[['quantity', 'sales_value']].sum()
    df_prices['price'] = df_prices.apply(lambda item: 
                                         item['sales_value'] / item['quantity']
                                         if item['quantity'] else 0, axis=1
                                        )
    df_prices.drop(columns=['quantity', 'sales_value'], inplace=True)
    
    return df_prices