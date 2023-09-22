import torch
import re
import os


def merge_datasets(datasets):
    
    train_count = 0
    val_count = 0
    new_train = []
    new_query = []
    new_gallery = []
    train_ind_to_dataset, query_ind_to_dataset, gal_ind_to_dataset = {}, {}, {}
    for i, dataset in enumerate(datasets):
        train_map = {}
        val_map = {}

        for t in dataset.train:
            train_ind_to_dataset[len(train_ind_to_dataset)] = i
            id_ = t[1]
            if id_ not in train_map:
                train_map[id_] = train_count
                train_count += 1
            new_train.append((t[0], train_map[id_], t[1], t[2]))

        for t in dataset.gallery:
            gal_ind_to_dataset[len(gal_ind_to_dataset)] = i
            id_ = t[1]
            if id_ not in val_map:
                val_map[id_] = val_count
                val_count += 1
            new_gallery.append((t[0], val_map[id_], t[1], t[2]))


        for t in dataset.query:
            query_ind_to_dataset[len(query_ind_to_dataset)] = i
            id_ = t[1]
            if id_ not in val_map:
                val_map[id_] = val_count
                val_count += 1
            new_query.append((t[0], val_map[id_], t[1], t[2]))

    return new_train, new_query, new_gallery, train_ind_to_dataset, query_ind_to_dataset, gal_ind_to_dataset
