import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, item_features, weighting=True):
                
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid,\
            self.itemid_to_id, self.userid_to_id = prepare_dicts(self.user_item_matrix)
        self.sparse_user_item = csr_matrix(user_item_matrix).tocsr()
        
        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = self.prepare_item_id_to_ctm(item_features)
        
        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        
    def get_recommendations(self, user, N=5):
        """Рекомендуем топ-N товаров"""
    
        res = [self.id_to_itemid[rec[0]] for rec in 
                    self.model.recommend(userid=self.userid_to_id[user], 
                                         user_items=self.sparse_user_item,   # на вход user-item matrix
                                         N=N, 
                                         filter_already_liked_items=False, 
                                         filter_items=[itemid_to_id[999999]],  # !!! 
                                         recalculate_user=True)]
        return res

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        # Не забывайте, что нужно учесть параметр filter_ctm
        top_n_user_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N).item_id.tolist()
        res = []
        
        if filter_ctm:
            # N=50, чтобы повысить шанс найти среди похожих товаров найти товар ctm
            for item_id in top_n_user_purchases:
                similar_items = self.model.similar_items(itemid_to_id[item_id], N=50)
                for sim_item in similar_items:
                    if self.item_id_to_ctm[id_to_itemid[sim_item[0]]] == 1:
                        res.append(sim_item[0])
                        break # Останавливаем цикл после 1го найденного товара и переходим к следующей покупке юзера
        
        else:
            for item_id in top_n_user_purchases:
                similar_item = self.model.similar_items(itemid_to_id[item_id], N=2)
                res.append(similar_item[1][0])
                
        res = [id_to_itemid[item] for item in res]
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N)
        similar_users = [self.id_to_userid[user[0]] for user in similar_users]
        
        similar_users_items = self.top_purchases[self.top_purchases['user_id'].isin(similar_users)]
        similar_users_items.sort_values('quantity', ascending=False, inplace=True)
        
        res = similar_users_items['item_id'].head(N).tolist()
        return res
    
    @staticmethod
    def prepare_matrix(data):
        """Подготавливает user-item матрицу"""
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity', # Можно пробоват ьдругие варианты
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    @staticmethod
    def prepare_item_id_to_ctm(item_features):
        all_ids = item_features['item_id'].tolist()
        ctm_ids = set(item_features[item_features['brand'] == 'Private'].item_id.tolist())
        
        item_id_to_ctm = {}
        
        for item_id in all_ids:
            item_id_to_ctm[item_id] = 1 if item_id in ctm_ids else 0
            
        return item_id_to_ctm
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model