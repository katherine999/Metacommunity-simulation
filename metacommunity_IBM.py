# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:09:31 2022

@author: JH_Lin
"""
import numpy as np
import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import re
import pandas as pd
import seaborn as sns
import time
###################################################################################################################################################
class habitat():
    def __init__(self, hab_name, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width):
        '''
        int num_env_types is the number of environment types in the habitat.
        env_types_name is the list of names of env_types.
        list mean_env_ls is the list of mean environment values; the len(mean_env_ls)=num_env_types.
        list var_env_ls is the list of variation of enviroment distribution in the habitat.
        int length is the length of the habitat.
        int width is the width of the habitat.
        int size is the the number of microsites within a habitat.
        '''
        self.name = hab_name
        self.num_env_types = num_env_types
        self.env_types_name = env_types_name
        self.mean_env_ls = mean_env_ls
        self.var_env_ls = var_env_ls
        self.length = length
        self.width = width
        self.size = length*width
        self.set = {}                     # self.data_set={} # to be improved
        self.indi_num = 0
        self.offspring_pool = []
        self.dormancy_pool = []
        self.species_category = {}
        self.occupied_site_pos_ls = []
        self.empty_site_pos_ls = [(i, j) for i in range(length) for j in range(width)]
        
        ####### to be improve #######
        self.dormancy_pool_max_size = 25
        #############################
        
        for index in range(0, len(mean_env_ls)):
            mean_e_index = self.mean_env_ls[index]
            var_e_index = self.var_env_ls[index]
            name_e_index = self.env_types_name[index]
            microsite_e_values = np.random.normal(loc=0, scale=var_e_index, size=(self.length, self.width)) + mean_e_index
            self.set[name_e_index] = microsite_e_values

        microsite_individuals = [[None for i in range(self.length)] for i in range(self.width)]
        self.set['microsite_individuals'] = microsite_individuals
        
    def __str__(self):
        return str(self.set)
    
    def add_individual(self, indi_object, len_id, wid_id):
       
        if self.set['microsite_individuals'][len_id][wid_id] != None:
            print('the microsite in the habitat is occupied.')
        else:
            self.set['microsite_individuals'][len_id][wid_id] = indi_object
            self.empty_site_pos_ls.remove((len_id, wid_id))
            self.occupied_site_pos_ls.append((len_id, wid_id))
            self.indi_num +=1

            if indi_object.species_id in self.species_category.keys():
                if indi_object.gender in self.species_category[indi_object.species_id].keys():
                    self.species_category[indi_object.species_id][indi_object.gender].append((len_id, wid_id))
                else:
                    self.species_category[indi_object.species_id][indi_object.gender] = [(len_id, wid_id)]             
            else:
                self.species_category[indi_object.species_id] = {indi_object.gender:[(len_id, wid_id)]}
                
    def del_individual(self, len_id, wid_id):
        if self.set['microsite_individuals'][len_id][wid_id] == None:
            print('the microsite in the habitat is empty.')
        else:
            indi_object = self.set['microsite_individuals'][len_id][wid_id]
            self.set['microsite_individuals'][len_id][wid_id] = None
            self.empty_site_pos_ls.append((len_id, wid_id))
            self.occupied_site_pos_ls.remove((len_id, wid_id))
            self.indi_num -=1 
            self.species_category[indi_object.species_id][indi_object.gender].remove((len_id, wid_id))
                                
    def hab_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        mean_pheno_val_ls = self.mean_env_ls
        species_id = 'sp%d'%(species_2_phenotype_ls.index(mean_pheno_val_ls)+1)
        
        for row in range(self.length):
            for col in range(self.width):
                if reproduce_mode == 'asexual': gender = 'female'
                if reproduce_mode == 'sexual': gender = random.sample(('male', 'female'), 1)[0]
                indi_object = individual(species_id=species_id, traits_num=traits_num, pheno_names_ls=pheno_names_ls, gender=gender)
                indi_object.random_init_indi(mean_pheno_val_ls, pheno_var_ls, geno_len_ls)
                self.add_individual(indi_object, row, col)
        return 0
        
    def get_microsite_env_val_ls(self, len_id, wid_id):
        ''' return a list of environment value of all the environment type in the order of env_types_name '''
        env_val_ls = []
        for env_name in self.env_types_name:
            env_val = self.set[env_name][len_id][wid_id]
            env_val_ls.append(env_val)
        return env_val_ls
    
    def survival_rate(self, d, phenotype_ls, env_val_ls, w = 0.5):
        '''
        d is the baseline death rate responding to the disturbance strength.
        phenotype_ls is a list of phenotype of each trait.
        env_val_ls is a list of environment value responding to the environment type.
        w is the width of the fitness function.
        '''
        survival_rate = (1-d)
        power = 0
        n = 0
        for index in range(len(phenotype_ls)):
            ei = phenotype_ls[index]               #individual phenotype of a trait 
            em = env_val_ls[index]                 #microsite environment value of a environment type
            power += math.pow(((ei-em)/w),2)
            n += 1
        survival_rate = (1-d) * math.exp((-1/n)*power)
        return survival_rate
    
    def hab_dead_selection(self, base_dead_rate, fitness_wid):
        counter = 0
        for row in range(self.length):
            for col in range(self.width):
                env_val_ls = self.get_microsite_env_val_ls(row, col)
                
                if self.set['microsite_individuals'][row][col] != None:
                    individual_object = self.set['microsite_individuals'][row][col]
                    phenotype_ls = individual_object.get_indi_phenotype_ls()
                    survival_rate = self.survival_rate(d=base_dead_rate, phenotype_ls=phenotype_ls, env_val_ls=env_val_ls, w = fitness_wid)
                    
                    if survival_rate < np.random.uniform(0,1,1)[0]:
                        self.del_individual(len_id=row, wid_id=col)
                        counter += 1
                    else:
                        continue
                else:
                    continue
        return counter

    def hab_asex_reproduce_mutate(self, birth_rate, mutation_rate, pheno_var_ls):
        self.offspring_pool = []
        nums = int(birth_rate)
        rate = birth_rate - nums
        for row in range(self.length):
            for col in range(self.width):
                if self.set['microsite_individuals'][row][col] == None:
                    continue
                else:
                    individual_object = self.set['microsite_individuals'][row][col]
                    for num in range(nums):
                        new_indivi_object = copy.deepcopy(individual_object)
                        for i in range(new_indivi_object.traits_num):
                            pheno_name = new_indivi_object.pheno_names_ls[i]
                            var = pheno_var_ls[i] #### to be improved ####
                            genotype = new_indivi_object.genotype_set[pheno_name]
                            phenotype = np.mean(genotype) + random.gauss(0, var)
                            new_indivi_object.phenotype_set[pheno_name] = phenotype
                        #print(individual_object, '\n'), print(new_indivi_object, '\n'), print('\n\n\n\n\n\n')
                        new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                        self.offspring_pool.append(new_indivi_object)
                        
                    if rate > np.random.uniform(0,1,1)[0]:
                        new_indivi_object = copy.deepcopy(individual_object)
                        for i in range(new_indivi_object.traits_num):
                            pheno_name = new_indivi_object.pheno_names_ls[i]
                            var = pheno_var_ls[i] #### to be improved ####
                            genotype = new_indivi_object.genotype_set[pheno_name]
                            phenotype = np.mean(genotype) + random.gauss(0, var)
                            new_indivi_object.phenotype_set[pheno_name] = phenotype
                        #print(individual_object, '\n'), print(new_indivi_object, '\n'), print('\n\n\n\n\n\n')
                        new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                        self.offspring_pool.append(new_indivi_object)
        return 0
    
    def hab_asex_reproduce_mutate_with_num(self, mutation_rate, pheno_var_ls, num):
        ''' asexual reproduction for dispersal controlled by the parameter, num '''
        hab_disp_pool = []
        parent_pos_ls = random.sample(self.occupied_site_pos_ls, num)
        
        for parent_pos in parent_pos_ls:
            row = parent_pos[0]
            col = parent_pos[1]
            individual_object = self.set['microsite_individuals'][row][col]
            new_indivi_object = copy.deepcopy(individual_object)
            for i in range(new_indivi_object.traits_num):
                pheno_name = new_indivi_object.pheno_names_ls[i]
                var = pheno_var_ls[i] #### to be improved #### 
                genotype = new_indivi_object.genotype_set[pheno_name]
                phenotype = np.mean(genotype) + random.gauss(0, var)
                new_indivi_object.phenotype_set[pheno_name] = phenotype
            new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            hab_disp_pool.append(new_indivi_object)
        return hab_disp_pool
    
    def hab_sex_reproduce_mutate_with_num(self, mutation_rate, pheno_var_ls, num):
        ''' asexual reproduction for dispersal controlled by the parameter, num '''
        hab_disp_pool = []
        pairwise_parents_pos_ls = random.sample(self.hab_sexual_pairwise_parents_ls(), num)
        
        for female_pos, male_pos in pairwise_parents_pos_ls:
            female_row, female_col = female_pos[0], female_pos[1]
            male_row, male_col = male_pos[0], male_pos[1]
            female_indi_obj = self.set['microsite_individuals'][female_row][female_col]
            male_indi_obj = self.set['microsite_individuals'][male_row][male_col]
            
            new_indivi_object = copy.deepcopy(female_indi_obj)
            new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
            for i in range(new_indivi_object.traits_num):
                pheno_name = new_indivi_object.pheno_names_ls[i]
                var = pheno_var_ls[i] ##### to be improved  #####
                
                female_bi_genotype = female_indi_obj.genotype_set[pheno_name]
                genotype1 = random.sample(female_bi_genotype, 1)[0]
                
                male_bi_genotype = male_indi_obj.genotype_set[pheno_name]
                genotype2 = random.sample(male_bi_genotype, 1)[0]
                
                new_bi_genotype = [genotype1, genotype2]
                phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                
                new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                new_indivi_object.phenotype_set[pheno_name] = phenotype
                
            new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            hab_disp_pool.append(new_indivi_object)
        return hab_disp_pool
             
    def hab_sex_reproduce_mutate(self, birth_rate, mutation_rate, pheno_var_ls):
        nums = int(birth_rate)
        rate = birth_rate - nums
        self.offspring_pool = []
        for sp_id, sp_id_val in self.species_category.items():
            try:
                sp_id_female_ls = sp_id_val['female']
            except:
                continue
            try:
                sp_id_male_ls = sp_id_val['male']
            except:
                continue
            
            random.shuffle(sp_id_female_ls) # list of individuals location in habitat, i.e., (len_id, wid_id)
            random.shuffle(sp_id_male_ls) # random sample of pairwise parents in sexual reproduction
            
            for female_pos, male_pos in list(zip(sp_id_female_ls, sp_id_male_ls)):
                female_indi_obj = self.set['microsite_individuals'][female_pos[0]][female_pos[1]]
                male_indi_obj = self.set['microsite_individuals'][male_pos[0]][male_pos[1]]
                for num in range(nums):
                    new_indivi_object = copy.deepcopy(female_indi_obj)
                    new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
                    for i in range(new_indivi_object.traits_num):
                        pheno_name = new_indivi_object.pheno_names_ls[i]
                        var = pheno_var_ls[i] ##### to be improved  #####

                        female_bi_genotype = female_indi_obj.genotype_set[pheno_name]
                        genotype1 = random.sample(female_bi_genotype, 1)[0]
                        
                        male_bi_genotype = male_indi_obj.genotype_set[pheno_name]
                        genotype2 = random.sample(male_bi_genotype, 1)[0]
                                               
                        new_bi_genotype = [genotype1, genotype2]
                        phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                        
                        new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                        new_indivi_object.phenotype_set[pheno_name] = phenotype
                    #print('female_indi_obj', female_indi_obj, '\n'), print('male_indi_obj', male_indi_obj, '\n'), print('new_indivi_object', new_indivi_object, '\n\n\n\n\n\n')    
                    new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                    self.offspring_pool.append(new_indivi_object)
                
                if rate > np.random.uniform(0,1,1)[0]:
                    new_indivi_object = copy.deepcopy(female_indi_obj)
                    new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
                    for i in range(new_indivi_object.traits_num):
                        pheno_name = new_indivi_object.pheno_names_ls[i]
                        var = pheno_var_ls[i] ##### to be improved  #####

                        female_bi_genotype = female_indi_obj.genotype_set[pheno_name]
                        genotype1 = random.sample(female_bi_genotype, 1)[0]
                        
                        male_bi_genotype = male_indi_obj.genotype_set[pheno_name]
                        genotype2 = random.sample(male_bi_genotype, 1)[0]
                                               
                        new_bi_genotype = [genotype1, genotype2]
                        phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                        
                        new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                        new_indivi_object.phenotype_set[pheno_name] = phenotype
                    #print('female_indi_obj', female_indi_obj, '\n'), print('male_indi_obj', male_indi_obj, '\n'), print('new_indivi_object', new_indivi_object, '\n\n\n\n\n\n')    
                    new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                    self.offspring_pool.append(new_indivi_object)
                else:
                    continue
        return 0                 
    
    def hab_sexual_pairwise_parents_ls(self):
        pair_parents_ls = []
        for sp_id, sp_id_val in self.species_category.items():
            try:
                sp_id_female_ls = sp_id_val['female']
            except:
                continue
            try:
                sp_id_male_ls = sp_id_val['male']
            except:
                continue
            
            random.shuffle(sp_id_female_ls) #list of individuals location in habitat, i.e., (len_id, wid_id)
            random.shuffle(sp_id_male_ls)   #random sample of pairwise parents in sexual reproduction
            
            pair_parents_ls += list(zip(sp_id_female_ls, sp_id_male_ls))
        return pair_parents_ls

    def hab_sexual_pairwise_parents_num(self):
        return len(self.hab_sexual_pairwise_parents_ls())

    def hab_germinate_from_offsprings_pool(self):
        ''' the offsprings in the habitat offsprings pool germinates in the empty microsite in the habitat'''
        counter = 0
        empty_sites_pos_ls = self.empty_site_pos_ls
        random.shuffle(empty_sites_pos_ls)
        
        hab_offsprings_pool = self.offspring_pool
        random.shuffle(hab_offsprings_pool)
        
        for pos, indi_object in list(zip(empty_sites_pos_ls, hab_offsprings_pool)):
            len_id = pos[0]
            wid_id = pos[1]
            self.add_individual(indi_object, len_id, wid_id)
            counter += 1
        return counter
    
    def hab_asexual_reprodece_germinate(self, birth_rate, mutation_rate, pheno_var_ls):
        empty_sites_pos_ls = self.empty_site_pos_ls
        if len(empty_sites_pos_ls) < int(self.indi_num * birth_rate): 
            num = len(empty_sites_pos_ls)
        elif len(empty_sites_pos_ls) >= int(self.indi_num * birth_rate): 
            num = int(self.indi_num * birth_rate)  
        hab_offsprings_for_germinate = self.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num)
        
        random.shuffle(empty_sites_pos_ls)
        random.shuffle(hab_offsprings_for_germinate)
        
        for pos, indi_object in list(zip(empty_sites_pos_ls, hab_offsprings_for_germinate)):
            len_id = pos[0]
            wid_id = pos[1]
            self.add_individual(indi_object, len_id, wid_id)
        return 0
        
    def hab_sexual_reprodece_germinate(self, birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        empty_sites_pos_ls = self.empty_site_pos_ls
        if len(empty_sites_pos_ls) < int(self.hab_sexual_pairwise_parents_num() * birth_rate): 
            num = len(empty_sites_pos_ls)
        elif len(empty_sites_pos_ls) >= int(self.hab_sexual_pairwise_parents_num() * birth_rate): 
            num = int(self.hab_sexual_pairwise_parents_num() * birth_rate)  
        hab_offsprings_for_germinate = self.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num)
    
        random.shuffle(empty_sites_pos_ls)
        random.shuffle(hab_offsprings_for_germinate)
        
        for pos, indi_object in list(zip(empty_sites_pos_ls, hab_offsprings_for_germinate)):
            len_id = pos[0]
            wid_id = pos[1]
            self.add_individual(indi_object, len_id, wid_id)
            counter += 1
        return counter
    
    def hab_dormancy_process(self):
        ''' offsprings pool into dormancy pool'''
        if len(self.offspring_pool) + len(self.dormancy_pool) <= self.dormancy_pool_max_size:
            self.dormancy_pool = self.dormancy_pool + self.offspring_pool
        else: 
            hab_dormancy_pool = random.sample(self.dormancy_pool,(self.dormancy_pool_max_size-len(self.offspring_pool)))
            self.dormancy_pool = hab_dormancy_pool + self.offspring_pool
            
        self.offspring_pool = []
        return 0
        
class patch():
    def __init__(self, patch_name, location, birth_rate):
        self.name = patch_name
        self.set = {}            # self.data_set={} # to be improved
        self.hab_num = 0
        self.location = location
        self.birth_rate = birth_rate
    def get_data(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.set
        return output

    def get_patch_microsites_individals_sp_id_values(self):
        ''' get species_id, phenotypes distribution in the patch as values set '''
        values_set = []
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            for row in range(hab_len):
                for col in range(hab_wid):
                    individual_object = h_object.set['microsite_individuals'][row][col]
                    if individual_object ==None:
                        values_set.append(np.nan)
                    else:
                        species_id = individual_object.species_id
                        species_id_value = int(re.findall(r"\d+",species_id)[0])
                        values_set.append(species_id_value)
        values_set = np.array(values_set).reshape(self.hab_num, h_object.size)
        return values_set
    
    def get_patch_microsites_individals_phenotype_values(self, trait_name):
        ''' get species_id, phenotypes distribution in the patch as values set '''
        values_set = []
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            for row in range(hab_len):
                for col in range(hab_wid):
                    individual_object = h_object.set['microsite_individuals'][row][col]
                    if individual_object ==None:
                        values_set.append(np.nan)
                    else:
                        phenotype = individual_object.phenotype_set[trait_name]
                        values_set.append(phenotype)
        values_set = np.array(values_set).reshape(self.hab_num, h_object.size)
        return values_set
    
    def __str__(self):
        return str(self.get_data())
    
    def get_patch_size(self):
        patch_size = 0
        for key, value in self.set.items():
            patch_size += value.size
        return patch_size
    
    def get_patch_individual_num(self):
        num = 0
        for key, value in self.set.items():
            num += value.indi_num
        return num
    
    def get_patch_sexual_pairwise_parents_num(self):
        num = 0
        for h_id, h_object in self.set.items():
            num += h_object.hab_sexual_pairwise_parents_num()
        return num
    
    def get_disp_within_offsprings_pool(self, target_hab_object):
        ''' return all the offspring (individual objects) in the patch, 
        in the exception of the target hab as a list for dispersal with patches'''
        disp_within_patch_offsprings_pool = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                disp_within_patch_offsprings_pool += h_object.offspring_pool
            else:
                continue
        return disp_within_patch_offsprings_pool
    
    def get_disp_within_asex_parent_pos_ls(self, target_hab_object):
        disp_within_patch_parent_pos_ls = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                occupied_site_pos_ls = h_object.occupied_site_pos_ls
                for site_pos in occupied_site_pos_ls:
                    site_pos = (h_id, ) + site_pos
                    disp_within_patch_parent_pos_ls.append(site_pos)
            else:
                continue
        return disp_within_patch_parent_pos_ls
    
    def get_disp_within_sex_pairwise_parents_pos_ls(self, target_hab_object):
        disp_within_patch_parent_pos_ls = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                pairwise_parents_pos_ls = h_object.hab_sexual_pairwise_parents_ls()
                for pairwise_parents_pos in pairwise_parents_pos_ls:
                    female_pos = pairwise_parents_pos[0]
                    male_pos = pairwise_parents_pos[1]
                    female_pos = (h_id, ) + female_pos
                    male_pos = (h_id, ) + male_pos
                    disp_within_patch_parent_pos_ls.append((female_pos, male_pos))
            else:
                continue
        return disp_within_patch_parent_pos_ls
    
    def get_patch_offsprings_pool(self):
        ''' return all the offspring (individual objects) in the patch as a list for dispersal among patches'''
        patch_offsprings_pool = []
        for h_id, h_object in self.set.items():
            patch_offsprings_pool += h_object.offspring_pool
        return patch_offsprings_pool
    
    def patch_offsprings_num(self):
        ''' return the number of offsprings in the patches '''
        return len(self.get_patch_offsprings_pool())
    
    def get_patch_empty_sites_ls(self):
        ''' return patch_empty_pos_ls as [(h_id, len_id, wid_id)] '''
        patch_empty_pos_ls = []
        for h_id, h_object in self.set.items():
            empty_site_pos_ls = h_object.empty_site_pos_ls
            for site_pos in empty_site_pos_ls:
                site_pos = (h_id, ) + site_pos
                patch_empty_pos_ls.append(site_pos)
        return patch_empty_pos_ls
    
    def patch_empty_sites_num(self):
        ''' return the number of empty microsite in the patches '''
        return len(self.get_patch_empty_sites_ls())
        
    def add_habitat(self, hab_name, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width):
        h_object = habitat(hab_name, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width)
        self.set[hab_name] = h_object
        self.hab_num += 1
        
    def patch_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for h_id, h_object in self.set.items():
            h_object.hab_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        return 0
    
    def patch_dead_selection(self, base_dead_rate, fitness_wid):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_dead_selection(base_dead_rate, fitness_wid)
        return counter
    
    def patch_asex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        birth_rate = self.birth_rate
        for h_id, h_object in self.set.items():
            h_object.hab_asex_reproduce_mutate(birth_rate, mutation_rate, pheno_var_ls)
        return 0
    
    def patch_sex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        birth_rate = self.birth_rate
        for h_id, h_object in self.set.items():
            h_object.hab_sex_reproduce_mutate(birth_rate, mutation_rate, pheno_var_ls)
        return 0
    
    def asex_reproduce_mutate_for_dispersal_among_patches(self, mutation_rate, pheno_var_ls, patch_offs_num):
        patch_disp_among_pool = []
        patch_indi_num = self.get_patch_individual_num()
        
        if patch_offs_num == 0 or patch_indi_num == 0:
            return patch_disp_among_pool
        else:
            for h_id, h_object in self.set.items():
                hab_offs_num = int(patch_offs_num * (h_object.indi_num/patch_indi_num))
                patch_disp_among_pool += h_object.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, hab_offs_num)
            return patch_disp_among_pool
    
    def sex_reproduce_mutate_for_dispersal_among_patches(self, mutation_rate, pheno_var_ls, patch_offs_num):
        patch_disp_among_pool = []
        patch_pairwise_parents_num = self.get_patch_sexual_pairwise_parents_num()
        if patch_offs_num == 0 or patch_pairwise_parents_num == 0:
            return patch_disp_among_pool
        else:
            for h_id, h_object in self.set.items():
                hab_offs_num = int(patch_offs_num * (h_object.hab_sexual_pairwise_parents_num()/patch_pairwise_parents_num))
                patch_disp_among_pool += h_object.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, hab_offs_num)
            return patch_disp_among_pool
    
    def patch_disp_within_from_offsprings_pool(self, disp_within_rate, counter):
        ''''''
        for h_id, h_object in self.set.items():
            h_empty_site_ls = h_object.empty_site_pos_ls
            disp_within_sites = random.sample(h_empty_site_ls, int(len(h_empty_site_ls)*disp_within_rate))
            disp_within_pool = self.get_disp_within_offsprings_pool(h_object)
            
            if len(disp_within_pool) > len(disp_within_sites):
                disp_within_indi_ls = random.sample(disp_within_pool, int(len(h_empty_site_ls)*disp_within_rate))
            else:
                disp_within_indi_ls = disp_within_pool
                random.shuffle(disp_within_indi_ls)
            
            for empty_site_pos, disp_indi_object in list(zip(disp_within_sites, disp_within_indi_ls)):
                len_id = empty_site_pos[0]
                wid_id = empty_site_pos[1]
                self.set[h_id].add_individual(indi_object=disp_indi_object, len_id=len_id, wid_id=wid_id)
                #print(counter, self.name, h_object.name, len_id, wid_id)
                counter += 1
        return counter
    
    def asex_reproduce_mutate_for_dispersal_within_patch(self, mutation_rate, pheno_var_ls, disp_within_rate, counter):
        ''''''
        birth_rate = self.birth_rate
        for h_id, h_object in self.set.items():
            h_empty_site_ls = h_object.empty_site_pos_ls
            asex_parent_pos_ls = self.get_disp_within_asex_parent_pos_ls(h_object)
            
            offsprings_expection_num = int(len(asex_parent_pos_ls)*birth_rate)
            disp_within_empty_site_num = int(len(h_empty_site_ls)*disp_within_rate)
            
            if offsprings_expection_num > disp_within_empty_site_num:
                disp_within_sites = random.sample(h_empty_site_ls, disp_within_empty_site_num)
                disp_within_parent_pos_ls = random.sample(asex_parent_pos_ls, disp_within_empty_site_num)
            else:
                disp_within_sites = random.sample(h_empty_site_ls, offsprings_expection_num)
                disp_within_parent_pos_ls = random.sample(asex_parent_pos_ls, offsprings_expection_num)
                
            for empty_site_pos, parent_pos in list(zip(disp_within_sites, disp_within_parent_pos_ls)):
                empty_len_id, empty_wid_id = empty_site_pos[0], empty_site_pos[1]
                parent_h_id, parent_row, parent_col = parent_pos[0], parent_pos[1], parent_pos[2]
                
                parent_indi_object = self.set[parent_h_id].set['microsite_individuals'][parent_row][parent_col]
                new_indivi_object = copy.deepcopy(parent_indi_object)
                
                for i in range(new_indivi_object.traits_num):
                    pheno_name = new_indivi_object.pheno_names_ls[i]
                    var = pheno_var_ls[i] #### to be improved #### 
                    genotype = new_indivi_object.genotype_set[pheno_name]
                    phenotype = np.mean(genotype) + random.gauss(0, var)
                    new_indivi_object.phenotype_set[pheno_name] = phenotype
                    new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                self.set[h_id].add_individual(indi_object=new_indivi_object, len_id=empty_len_id, wid_id=empty_wid_id)
                counter += 1
        return counter
            
    def sex_reproduce_mutate_for_dispersal_within_patch(self, mutation_rate, pheno_var_ls, disp_within_rate):
        ''''''
        birth_rate = self.birth_rate
        counter = 0
        for h_id, h_object in self.set.items():
            h_empty_site_ls = h_object.empty_site_pos_ls
            sex_pairwise_parents_pos_ls = self.get_disp_within_sex_pairwise_parents_pos_ls(h_object)
            
            offsprings_expection_num = int(len(sex_pairwise_parents_pos_ls)*birth_rate)
            disp_within_empty_site_num = int(len(h_empty_site_ls)*disp_within_rate)
            
            if offsprings_expection_num > disp_within_empty_site_num:
                disp_within_sites = random.sample(h_empty_site_ls, disp_within_empty_site_num)
                disp_within_pairwise_parent_pos_ls = random.sample(sex_pairwise_parents_pos_ls, disp_within_empty_site_num)
            else:
                disp_within_sites = random.sample(h_empty_site_ls, offsprings_expection_num)
                disp_within_pairwise_parent_pos_ls = random.sample(sex_pairwise_parents_pos_ls, offsprings_expection_num)
            
            for empty_site_pos, pairwise_parents_pos in list(zip(disp_within_sites, disp_within_pairwise_parent_pos_ls)):
                empty_len_id, empty_wid_id = empty_site_pos[0], empty_site_pos[1]
                female_parent_h_id, female_parent_row, female_parent_col = pairwise_parents_pos[0][0], pairwise_parents_pos[0][1], pairwise_parents_pos[0][2]
                male_parent_h_id, male_parent_row, male_parent_col = pairwise_parents_pos[1][0], pairwise_parents_pos[1][1], pairwise_parents_pos[1][2]
                
                female_parent_indi_object = self.set[female_parent_h_id].set['microsite_individuals'][female_parent_row][female_parent_col]
                male_parent_indi_object = self.set[male_parent_h_id].set['microsite_individuals'][male_parent_row][male_parent_col]
                
                new_indivi_object = copy.deepcopy(female_parent_indi_object)
                new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
                for i in range(new_indivi_object.traits_num):
                    pheno_name = new_indivi_object.pheno_names_ls[i]
                    var = pheno_var_ls[i] # to be improved
                    
                    female_bi_genotype = female_parent_indi_object.genotype_set[pheno_name]
                    genotype1 = random.sample(female_bi_genotype, 1)[0]
                    
                    male_bi_genotype = male_parent_indi_object.genotype_set[pheno_name]
                    genotype2 = random.sample(male_bi_genotype, 1)[0]
                    
                    new_bi_genotype = [genotype1, genotype2]
                    phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                    
                    new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                    new_indivi_object.phenotype_set[pheno_name] = phenotype
                new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                self.set[h_id].add_individual(indi_object=new_indivi_object, len_id=empty_len_id, wid_id=empty_wid_id)
                counter += 1
        return counter
    
    def patch_dormancy_processes(self):
        for h_id, h_object in self.set.items():
            h_object.hab_dormancy_process()
        return 0
    
    def patch_germinate_from_offsprings_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_germinate_from_offsprings_pool()
        return counter
    
    def patch_asexual_birth_germinate(self, mutation_rate, pheno_var_ls):
        birth_rate = self.birth_rate
        for h_id, h_object in self.set.items():
            h_object.hab_asexual_reprodece_germinate(birth_rate, mutation_rate, pheno_var_ls)
        return 0
    
    def patch_sexual_birth_germinate(self, mutation_rate, pheno_var_ls):
        birth_rate = self.birth_rate
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sexual_reprodece_germinate(birth_rate, mutation_rate, pheno_var_ls)
        #print(self.name, counter)
        return counter
    
class metacommunity():
    def __init__(self, metacommunity_name):
        self.set = {}                       # self.data_set={} # to be improved
        self.patch_num = 0
        self.meta_map = nx.Graph()
        self.metacommunity_name = metacommunity_name
    
    def get_data(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.get_data()
        return output
    
    def __str__(self):
        return str(self.get_data())
        
    def add_patch(self, patch_name, patch_object):
        ''' add new patch to the metacommunity. '''
        self.set[patch_name] = patch_object
        self.patch_num += 1
        self.meta_map.add_node(patch_name)
        
    def get_all_patches_location(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.location
        return output
    
    def get_meta_individual_num(self):
        num = 0
        for patch_id, patch_object in self.set.items():
            num += patch_object.get_patch_individual_num()
        return num
    
    def show_meta_individual_num(self):
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(indi_num, empty_sites_num))
        return 0
    
    def show_meta_map(self, graph_object):
        pos = self.get_all_patches_location()
        plt.figure(figsize=(10,8))
        nx.draw_networkx(graph_object, pos=pos)
        nx.draw_networkx(graph_object, pos=pos, edge_color='b') 
        plt.savefig('meta_network.jpg')
        return 0
    
    ### to be improved, the coordination of the metacommunity ###
    def show_meta_species_distribution(self, cmap, file_name):
        fig = plt.figure(figsize=(200, 160))
        plt.title('species distribution across metacommunity')
        
        for patch_id, patch_object in self.set.items():
            patch_location = patch_object.location
            l = (9-patch_location[1])*10 + (1+patch_location[0])
            
            ax = fig.add_subplot(10, 10, l)
            ax.set_title(patch_id + ',n='+ str(int(patch_object.get_patch_size()/patch_object.hab_num)), fontsize = 170)
            plt.tight_layout()

            df = pd.DataFrame(patch_object.get_patch_microsites_individals_sp_id_values())
            sns.heatmap(data=df, vmin=1, vmax=30, cmap=cmap, yticklabels=False, cbar=False)
           
        plt.savefig(file_name)
        plt.clf()
        return 0

    #### to be improved ####
    def customize_meta_map(self):
        pass
    ########################
    
    def full_con_map(self):
        ''' return full connected metacommunity network '''
        full_con_map = nx.Graph()
        pos = self.get_all_patches_location() # localtion of patch in dir
        for key1, value1 in pos.items():
            for key2, value2 in pos.items():
                if key1 != key2:
                    distance = math.sqrt(math.pow((value1[0]-value2[0]), 2)+math.pow((value1[1]-value2[1]), 2))
                    #print(key1, value1, key2, value2, distance)
                    full_con_map.add_edge(key1, key2, weight = distance)
                    # add all posible edges to form Graph with full connectance
        return full_con_map
                    
    def mini_span_tree(self):
        ''' return minimum connected metacommunity network '''
        full_con_map = self.full_con_map()
        mini_map = nx.minimum_spanning_tree(full_con_map)
        return mini_map
        
    def med_con_map(self, add_links_propotion):
        ''' return medimum connected metacommunity network with the degree of 'add_links_propotion'. '''
        full_con_map = self.full_con_map()
        mini_con_map = self.mini_span_tree()
        med_con_map = mini_con_map
        all_add_links = set(full_con_map.edges())-set(mini_con_map.edges())
        add_links = random.sample(all_add_links, int(len(all_add_links)*add_links_propotion))
        pos = self.get_all_patches_location()
        for edge in add_links:
            patch_id1, patch_id2 = edge[0], edge[1]
            location1, location2 = pos[patch_id1], pos[patch_id2]
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            med_con_map.add_edge(patch_id1, patch_id2, weight = distance)
        return med_con_map
    
    def dist2disp_function(self, k, x):
        ''' Exponential decay model for dispersal among patches.
        k is a scaling factor determining the strength of dispersal limitation.
        x is the distance between the two patches. '''
        return k * np.exp(-k*x)
    
    def mat_around(self, matrix):
        for i in range(matrix.shape[0]):
            if 0<matrix[i, i]<1:
                matrix[i,i]=1
        return np.around(matrix)
    
    def emigrant_disp_rate_matrix(self, disp_kernal, graph_object):
        ''' return emigrant_dispersal_rate_matrix.
        the elements D_ij (row_i, col_j) in the matric means the probability that 
        the emigrants to patch j of the all emigrants (offspings) from patch i.
        the row vector of the matrix is idendity vector. '''
        #dis_matrix = nx.adjacency_matrix(graph_object).todense()
        short_path_dis_matrix = nx.floyd_warshall_numpy(graph_object)
        disp_kernal_matrix = self.dist2disp_function(disp_kernal, short_path_dis_matrix)
        disp_rate_matrix = disp_kernal_matrix/disp_kernal_matrix.sum(axis=1)   #normalization
        # column vector of disp_rate_matrix is the dispersal rate leaving the patch. sum of column vector is 1.
        return disp_rate_matrix
    
    def emigrant_matrix_from_offsprings_pool(self, disp_kernal, graph_object):
        ''' 
        patch_offs_num_matrix is a diagonal matrix, the nonzero elements in which represent the num of offspring in each patch
        emigrant_disp_rate_matrix is emigrant_dispersal_rate_matrix.
        emigrant_matrix = patch_offs_num_matrix * emigrant_disp_rate_matrix
        the element(i,j) in the result matrix means, of all the emigrants patch i can provided, the nums of emigrants offsprings from patch i to patch j
        sum of row vector (i) is the num of all the offsprings in patch i
        '''
        patch_offs_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_off_num = patch_object.patch_offsprings_num()
            patch_offs_num_matrix[index, index] = patch_off_num
            index+=1
        return self.mat_around(patch_offs_num_matrix * self.emigrant_disp_rate_matrix(disp_kernal, graph_object))
    
    def emigrant_matrix_expectation_asexual(self, disp_kernal, graph_object):
        ''''''
        patch_offs_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_off_num = patch_object.get_patch_individual_num() * patch_object.birth_rate
            patch_offs_num_matrix[index, index] = patch_off_num
            index+=1
        return self.mat_around(patch_offs_num_matrix * self.emigrant_disp_rate_matrix(disp_kernal, graph_object))
    
    def emigrant_matrix_expectation_sexual(self, disp_kernal, graph_object):
        ''''''
        patch_offs_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_off_num = patch_object.get_patch_sexual_pairwise_parents_num() * patch_object.birth_rate
            patch_offs_num_matrix[index, index] = patch_off_num
            index+=1
        return self.mat_around(patch_offs_num_matrix * self.emigrant_disp_rate_matrix(disp_kernal, graph_object))
    
    def immigrant_disp_rate_matrix(self, disp_kernal, graph_object):
        ''' return imigrant_dispersal_rate_matrix.
        the elements D_ij (row_i, col_j) in the matric means the probability that 
        the immigarnt to patch j from patch i over all the immigarnt to patch j 
        including all the others patches besides patch i
        the col vector of the matrix is idendity vector.
        '''
        short_path_dis_matrix = nx.floyd_warshall_numpy(graph_object)
        disp_kernal_matrix = self.dist2disp_function(disp_kernal, short_path_dis_matrix)
        disp_rate_matrix = disp_kernal_matrix/disp_kernal_matrix.sum(axis=0)   #normalization
        # column vector of disp_rate_matrix is the dispersal rate leaving the patch. sum of column vector is 1.
        return disp_rate_matrix
    
    def immigrant_matrix_to_patch_empty_sites(self, disp_kernal, graph_object):
        '''
        patch_empty_sites_num_matrix is a diagonal matrix, the nonzero elements in which represent the num of empty sites in each patch
        imigrant_dispersal_rate_matrix is imigrant_dispersal_rate_matrix
        immigrant_matrix = imigrant_dispersal_rate_matrix * patch_empty_sites_num_matrix
        the element(i,j) in the result matrix means, of all the empty sites in patch j, the num of immigarnts from patch i
        sum of column vector (j) is the num of all the empty site in patch j 
        '''
        patch_empty_sites_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_empty_sites_num = patch_object.patch_empty_sites_num()
            patch_empty_sites_num_matrix[index, index] = patch_empty_sites_num
            index += 1
        return self.mat_around(self.immigrant_disp_rate_matrix(disp_kernal, graph_object) * patch_empty_sites_num_matrix)
    
    def meta_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        return 0
    
    def get_meta_empty_sites_ls(self):
        ''' return meta_empty_sites_ls as [(patch_id, h_id, len_id, wid_id)] '''   
        meta_empty_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_empty_pos_ls = patch_object.get_patch_empty_sites_ls()
            for empty_pos in patch_empty_pos_ls:
                empty_pos = (patch_id, ) + empty_pos
                meta_empty_sites_ls.append(empty_pos)
        return meta_empty_sites_ls
    
    def show_meta_empty_sites_num(self):
        return len(self.get_meta_empty_sites_ls())
    
    def meta_dead_selection(self, base_dead_rate, fitness_wid):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dead_selection(base_dead_rate, fitness_wid)
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals dead in selection; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_asex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_asex_reproduce_mutate(mutation_rate, pheno_var_ls)
        return 0
    
    def meta_sex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_sex_reproduce_mutate(mutation_rate, pheno_var_ls)
        return 0
    
    def meta_disp_among_patches_from_offsprings_pool(self, disp_kernal, graph_object):
        ''' dispersal from patch_i to patch j'''
        emigrants_matrix = self.emigrant_matrix_from_offsprings_pool(disp_kernal, graph_object)
        immigrants_matrix = self.immigrant_matrix_to_patch_empty_sites(disp_kernal, graph_object)
        migrants_matrix = np.minimum(emigrants_matrix, immigrants_matrix)
        counter = 0
        # dispersal from patch i to patch j
        for j in range(len(self.meta_map.nodes())):
            patch_j_id = list(self.meta_map.nodes())[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_indi_object_ls = []
            
            for i in range(len(self.meta_map.nodes())):
                patch_i_id = list(self.meta_map.nodes())[i]
                patch_i_object = self.set[patch_i_id]
                patch_i_offspring_pool = patch_i_object.get_patch_offsprings_pool()
            
                if i==j: 
                    continue
                else:
                    migrants_num = int(migrants_matrix[i, j])
                    migrants_indi_object_ls += random.sample(patch_i_offspring_pool, migrants_num)
                    
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_indi_object_ls)
            
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                #print(counter, patch_j_id, h_id, len_id, wid_id)  
                counter += 1
                
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_disp_among_patches_from_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_disp_among_patches_from_offsprings_and_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_disp_within_patch_from_offsprings_pool(self, disp_within_rate):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter = patch_object.patch_disp_within_from_offsprings_pool(disp_within_rate, counter)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals disperse within patch; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_disp_within_patches_from_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_disp_within_patches_from_offsprings_and_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_asexual_reproduce_mutate_and_dispersal_among_patches(self, mutation_rate, pheno_var_ls, disp_kernal, graph_object):
        ''' dispersal from patch_i to patch j'''
        emigrants_matrix = self.emigrant_matrix_expectation_asexual(disp_kernal, graph_object)
        immigrants_matrix = self.immigrant_matrix_to_patch_empty_sites(disp_kernal, graph_object)
        migrants_matrix = np.minimum(emigrants_matrix, immigrants_matrix)
        counter = 0
        for j in range(len(self.meta_map.nodes())):
            patch_j_id = list(self.meta_map.nodes())[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_indi_object_ls = []
            
            for i in range(len(self.meta_map.nodes())):
                patch_i_id = list(self.meta_map.nodes())[i]
                patch_i_object = self.set[patch_i_id]
                if i==j: 
                    continue
                else:
                    migrants_num = int(migrants_matrix[i, j])
                    patch_i_offspring_pool = patch_i_object.asex_reproduce_mutate_for_dispersal_among_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, patch_offs_num=migrants_num)
                    migrants_indi_object_ls += patch_i_offspring_pool
                    
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_indi_object_ls)
            
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                #print(counter, patch_j_id, h_id, len_id, wid_id)  
                counter += 1
        print('there are %d individuals disperse among patches'%counter)
    
    def meta_sexual_reproduce_mutate_and_dispersal_among_patches(self, mutation_rate, pheno_var_ls, disp_kernal, graph_object):
        emigrants_matrix =  self.emigrant_matrix_expectation_sexual(disp_kernal, graph_object)
        immigrants_matrix = self.immigrant_matrix_to_patch_empty_sites(disp_kernal, graph_object)
        migrants_matrix = np.minimum(emigrants_matrix, immigrants_matrix)
        counter = 0
        for j in range(len(self.meta_map.nodes())):
            patch_j_id = list(self.meta_map.nodes())[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_indi_object_ls = []
            
            for i in range(len(self.meta_map.nodes())):
                patch_i_id = list(self.meta_map.nodes())[i]
                patch_i_object = self.set[patch_i_id]
                if i==j: 
                    continue
                else:
                    migrants_num = int(migrants_matrix[i, j])
                    patch_i_offspring_pool = patch_i_object.sex_reproduce_mutate_for_dispersal_among_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, patch_offs_num=migrants_num)
                    migrants_indi_object_ls += patch_i_offspring_pool
                    
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_indi_object_ls)
            
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                #print(counter, patch_j_id, h_id, len_id, wid_id)  
                counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_asexual_birth_disp_within_patches(self, mutation_rate, pheno_var_ls, disp_within_rate):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter = patch_object.asex_reproduce_mutate_for_dispersal_within_patch(mutation_rate, pheno_var_ls, disp_within_rate, counter)
        print('there are %d individuals disperse within patch'%counter)
        return 0
    
    def meta_sexual_birth_disp_within_patches(self, mutation_rate, pheno_var_ls, disp_within_rate):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.sex_reproduce_mutate_for_dispersal_within_patch(mutation_rate, pheno_var_ls, disp_within_rate)
    
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals disperse within patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0

    def meta_germinate_from_offsprings_pool(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_germinate_from_offsprings_pool()
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals germinating from local offsprings pool; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_asexual_birth_mutate_germinate(self, mutation_rate, pheno_var_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_asexual_birth_germinate(mutation_rate, pheno_var_ls)
        return 0
    
    def meta_sexual_birth_mutate_germinate(self, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sexual_birth_germinate(mutation_rate, pheno_var_ls)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals germinating from local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
####################################################################################################################################################
class species_pool():
    def __init__(self, species_num, standar_species_ls):
        self.species_num = species_num
        self.standar_species_ls = standar_species_ls
    
    def generate_propagules_rain_ls(self, num, pheno_var_ls, geno_len_ls):
        propagules_rain_ls = []
        for i in range(num):
            standar_species_object = random.sample(self.standar_species_ls, 1)[0]
            individual_object = individual(species_id=standar_species_object.species_id, traits_num=standar_species_object.traits_num, pheno_names_ls=standar_species_object.pheno_names_ls)
            individual_object.random_init_indi(mean_pheno_val_ls=standar_species_object.mean_pheno_val_ls, pheno_var_ls=pheno_var_ls, geno_len_ls=geno_len_ls)
            propagules_rain_ls.append(individual_object)
        return propagules_rain_ls
    
####################################################################################################################################################    
class species():
    def __init__(self, species_id, traits_num, pheno_names_ls, mean_pheno_val_ls):

        self.species_id = species_id
        self.traits_num = traits_num
        self.pheno_names_ls = pheno_names_ls
        self.mean_pheno_val_ls = mean_pheno_val_ls
        
class individual():
    def __init__(self, species_id, traits_num, pheno_names_ls, gender='female', genotype_set=None, phenotype_set=None):
        self.species_id = species_id
        self.gender = gender
        self.traits_num = traits_num
        self.pheno_names_ls = pheno_names_ls
        self.genotype_set = genotype_set
        self.phenotype_set = phenotype_set
        
    def random_init_indi(self, mean_pheno_val_ls, pheno_var_ls, geno_len_ls):
        '''
        pheno_names is a tuple of the pheno_names (string) i.e., ('phenotye_1', 'phenotype_2',...,'phenotye_x') and the len(pheno_names) is equal to traits_num.
        mean_pheno_val (tuple) is the mean values (float) of the phenotypes of a species population which fit a gaussian distribution, i.e., (val1, val2,...,valx).
        pheno_var (tuple) is the variation (float) of the phenotypes of a species population which fit a gaussian distribution.
        geno_len (tuple) is the len of each genotype in the genotype_set, the genotype in which controls each phenotype of each trait.
        '''
        genotype_set = {}
        phenotype_set = {}
        
        for i in range(self.traits_num):
            name = self.pheno_names_ls[i]
            mean = mean_pheno_val_ls[i]
            var = pheno_var_ls[i]
            geno_len = geno_len_ls[i]
            
            #random_index = random.sample(range(0,geno_len*2),int(mean*geno_len*2))
            #genotype = np.array([1 if i in random_index else 0 for i in range(geno_len*2)])
            #bi_genotype = [genotype[0:geno_len], genotype[geno_len:geno_len*2]]
            
            random_index_1 = random.sample(range(0,geno_len),int(mean*geno_len))
            random_index_2 = random.sample(range(0,geno_len),int(mean*geno_len))
            genotype_1 = np.array([1 if i in random_index_1 else 0 for i in range(geno_len)])
            genotype_2 = np.array([1 if i in random_index_2 else 0 for i in range(geno_len)])
            
            bi_genotype = [genotype_1, genotype_2]
            phenotype = mean + random.gauss(0, var)
            
            genotype_set[name] = bi_genotype
            phenotype_set[name] = phenotype
        self.genotype_set = genotype_set
        self.phenotype_set = phenotype_set
        return 0
    
    def __str__(self):
        species_id_str = 'speceis_id=%s'%self.species_id
        gender_str = 'gender=%s'%self.gender
        traits_num_str = 'traits_num=%d'%self.traits_num
        genotype_set_str = 'genetype_set=%s'%str(self.genotype_set)
        phenotype_set_str = 'phenotype_set=%s'%str(self.phenotype_set)
        
        strings = species_id_str+'\n'+ gender_str+'\n'+traits_num_str+'\n'+genotype_set_str+'\n'+phenotype_set_str
        return strings
    
    def get_indi_phenotype_ls(self):
        indi_phenotype_ls = []
        for pheno_name in self.pheno_names_ls:
            phenotype = self.phenotype_set[pheno_name]
            indi_phenotype_ls.append(phenotype)
        return indi_phenotype_ls
    
    def mutation(self, rate, pheno_var_ls):
        for i in range(self.traits_num):
            mutation_counter = 0
            pheno_name = self.pheno_names_ls[i]
            var = pheno_var_ls[i]
            genotype1 = self.genotype_set[pheno_name][0]
            genotype2 = self.genotype_set[pheno_name][1]
            for index in range(len(genotype1)):
                if rate > np.random.uniform(0,1,1)[0]:
                    mutation_counter += 1
                    if genotype1[index] == 0: self.genotype_set[pheno_name][0][index]=1
                    elif genotype1[index] == 1: self.genotype_set[pheno_name][0][index]=0
                    
            for index in range(len(genotype2)):
                if rate > np.random.uniform(0,1,1)[0]:
                    mutation_counter += 1
                    if genotype2[index] == 0: self.genotype_set[pheno_name][1][index]=1
                    elif genotype2[index] == 1: self.genotype_set[pheno_name][1][index]=0
            if mutation_counter >=1: 
                phenotype = np.mean(self.genotype_set[pheno_name]) + random.gauss(0, var)
                self.phenotype_set[pheno_name] = phenotype
        return 0
            
###################################################################################################

if __name__ == '__main__':
    
    ####################
    patch_num = 25
    metacommunity_x_range = range(0,10)
    metacommunity_y_range = range(0,10)
    
    environment_types_num=2
    environment_types_name=('micro_environment', 'macro_environment')
    environment_variation_ls=(0.025, 0.025)
    micro_environment_values_ls = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    macro_environment_values_ls = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    #hab_num_per_patch_ls = [1,2,3,4,5,6]
    hab_num_per_patch_ls = [6]
    hab_sizes_ls = [(10,10), (20,20), (30,30)] 
    
    species_num = 30
    traits_num = 2
    pheno_names_ls = ('micro_phenotype', 'macro_phenotype')
    species_2_phenotype_ls = [(np.around(j, 1), np.around(i, 1)) for i in np.arange(0.1, 1.1, 0.2) for j in np.arange(0.0, 1.1, 0.2)]
    standar_species_object_ls = [species(species_id='sp%d'%(i+1), traits_num=traits_num, pheno_names_ls=pheno_names_ls, mean_pheno_val_ls=(species_2_phenotype_ls[i])) for i in range(species_num)]
    
    birth_rate = 1
    ####################
    mainland = species_pool(species_num=species_num, standar_species_ls=standar_species_object_ls)
    
    m1 = metacommunity(metacommunity_name='m1')
    patch_location_ls = random.sample(list(itertools.product(metacommunity_x_range, metacommunity_y_range)), patch_num)
    patch_macro_environment_means_values = [random.sample(macro_environment_values_ls, 1)[0] for i in range(patch_num)]
    
    for i in range(0, patch_num):
        patch_name = 'patch%s'%str(i+1)
        location = patch_location_ls[i]
        p = patch(patch_name, location, birth_rate)
        
        hab_num = random.sample(hab_num_per_patch_ls, 1)[0]
        hab_lenth, hab_width = random.sample(hab_sizes_ls, 1)[0]
        #hab_lenth, hab_width = 10, 10
        micro_environment_means_values_ls = random.sample(micro_environment_values_ls, hab_num)
        micro_environment_means_values_ls.sort()
        macro_environment_means_value = macro_environment_values_ls[(location[1])//2]
        
        for j in range(hab_num):
            habitat_name = 'h%s'%str(j+1)
            micro_environment_mean_value = micro_environment_means_values_ls[j]
            p.add_habitat(hab_name=habitat_name, num_env_types=environment_types_num, env_types_name=environment_types_name, 
                          mean_env_ls=(micro_environment_mean_value, macro_environment_means_value), var_env_ls=environment_variation_ls, length=hab_lenth, width=hab_width)
            print(patch_name, location, habitat_name, 'micro_environment_mean_value =', micro_environment_mean_value, ' macro_environment_means_value =', macro_environment_means_value)
        m1.add_patch(patch_name=patch_name, patch_object=p)
  
    mini = m1.mini_span_tree()  
    
    #m1.meta_initialize(traits_num=2, pheno_names_ls=('micro_phenotype', 'macro_phenotype'), pheno_var_ls=(0.025, 0.025), geno_len_ls=(20, 20), reproduce_mode='asexual', species_2_phenotype_ls=species_2_phenotype_ls)
    m1.meta_initialize(traits_num=2, pheno_names_ls=('micro_phenotype', 'macro_phenotype'), pheno_var_ls=(0.025, 0.025), geno_len_ls=(20, 20), reproduce_mode='sexual', species_2_phenotype_ls=species_2_phenotype_ls)
    
    starttime = time.time()
    for timp_step in range(1):
        d1 = time.time()
        print('time_step%d'%timp_step)
        '''
        m1.show_meta_individual_num()
        m1.meta_dead_selection(base_dead_rate=0.1, fitness_wid=0.5)
        m1.meta_asexual_reproduce_mutate_and_dispersal_among_patches(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025), disp_kernal=2, graph_object=mini)
        m1.meta_asexual_birth_disp_within_patches(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025), disp_within_rate=0.1)
        m1.meta_asexual_birth_mutate_germinate(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025))
        '''
        
        
        m1.show_meta_individual_num()
        m1.meta_dead_selection(base_dead_rate=0.1, fitness_wid=0.5)
        m1.meta_sexual_reproduce_mutate_and_dispersal_among_patches(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025), disp_kernal=3, graph_object=mini)
        m1.meta_sexual_birth_disp_within_patches(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025), disp_within_rate=0.1)
        m1.meta_sexual_birth_mutate_germinate(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025))
        
        
        '''
        m1.show_meta_individual_num()
        m1.meta_dead_selection(base_dead_rate=0.1, fitness_wid=0.5)
        m1.meta_asex_reproduce_mutate(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025))
        #m1.meta_sex_reproduce_mutate(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025))
        m1.meta_disp_among_patches_from_offsprings_pool(disp_kernal=2, graph_object=mini)
        m1.meta_disp_within_patch_from_offsprings_pool(disp_within_rate=0.1)
        m1.meta_germinate_from_offsprings_pool()
        '''
        d2 = time.time()
        print("程序运行时间：%.8s s" % (d2-d1)) 
        
    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)  
        
    cmap = cmap = sns.color_palette("Reds",7)[1:7] + sns.color_palette("Greens",7)[1:7]+sns.color_palette("Blues",7)[1:7]+sns.color_palette("Purples",7)[1:7]+sns.color_palette("Greys",7)[1:7]
    m1.show_meta_species_distribution(cmap, '5000sex_exp.jpg')
    
    























