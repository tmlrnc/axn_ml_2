import argparse
import csv
import math
import random




class DataSet(object):


    def __init__(self, filename):

        with open(filename, encoding='utf-8-sig') as io:
            self.records = list(csv.DictReader(io))
        self.record_count = 0
        for record in self.records:
            self.record_count = self.record_count + 1
            for k, v in record.items():
                feature = Feature.get_feature(k)
                feature.count(v)
        Feature.record_count = self.record_count


        return

    def get_smallest_sample_feature(self, noise_threshold, non_noisy_feature):
        random.shuffle(self.records)
        new_records = []
        category_count = {}
        metadata = non_noisy_feature.get_smallest_sample_metadata(noise_threshold)
        for category in metadata:
            category_count[category] = metadata[category]
        print(category_count)
        for r in self.records:
            target_category = str(r[non_noisy_feature.name])
            if target_category not in category_count:
                # If the target_category is not in our category_count, we skip this record
                continue
            # Otherwise, we save the record
            new_records.append(r)
            # And reduce the count
            category_count[target_category] -= 1
            if category_count[target_category] == 0:
                # If the target category count is 0, we are done and remove it from the dictionary
                del category_count[target_category]
            if category_count == {}:
                # If there are no more categories, we are done
                return new_records

        return new_records

    def write_smallest_sample_feature_to_csv(self,list_of_sample_for_feature, file_out_name,feature):

        new_end = "_" + str(feature) + ".csv"
        file_out_name_new = file_out_name.replace(".csv", new_end)
        with open(file_out_name_new, 'w', newline='') as csvfile:
            mywriter = csv.writer(csvfile)
            key_count = 0
            for row in list_of_sample_for_feature:
                if key_count == 0 :
                    list_keys = [k for k in row]
                    mywriter.writerow(list_keys)
                    key_count = 1
                list_values = [v for v in row.values()]
                mywriter.writerow(list_values)

    def get_non_noisy_features(self, noise_threshold):
        return Feature.get_non_noisy_features(noise_threshold)


class Category(object):

    def __init__(self, name, feature):
        self.name = name
        self.feature = feature
        self.count = 0
        self.sample_size = 0


class Feature(object):

    features = {}

    @staticmethod
    def get_feature(name):
        if(name in Feature.features):
            return Feature.features[name]
        return Feature(name)


    @staticmethod
    def get_non_noisy_features(noise_threshold):
        non_noisy = set()
        return set(f for f in Feature.features.values() if f.is_not_noisy(noise_threshold))

    @staticmethod
    def get_noisy_features(noise_threshold):
        noisy = set()
        return set(f for f in Feature.features.values() if f.is_noisy(noise_threshold))

    def __init__(self, name):
        if(name in Feature.features):
            raise Exception(f"Found two features with the same name. Ooops. {name}")
        self.name = name
        self.total_count = 0
        self.record_count = 0
        self.categories = {}
        Feature.features[name] = self

    def get_categories(self):
        return list(self.categories.values())


    @staticmethod
    def get_all_sample_sizes(noise_threshold):

        """

        percentage of smallest non noisy category = noise threshold / (SAMPLE SIZE I AM SEEKING)

        SAMPLE SIZE I AM SEEKING =  noise threshold / percentage of smallest non noisy category

        """

        record_cnt= Feature.record_count

        non_noisy_features = Feature.get_non_noisy_features(noise_threshold)
        for my_non_noisy_features in non_noisy_features:
            categories_values = list(my_non_noisy_features.categories.values())
            category_percentage = categories_values[0].count/record_cnt
            category_sample_size = round(noise_threshold/category_percentage)
            categories_values[0].sample_size = category_sample_size

        return

    @staticmethod
    def get_smallest_sample_size(noise_threshold):

        """

        percentage of smallest non noisy category = noise threshold / (SAMPLE SIZE I AM SEEKING)

        SAMPLE SIZE I AM SEEKING =  noise threshold / percentage of smallest non noisy category

        """

        categories_count = []
        total_count = 0
        record_cnt= Feature.record_count
        non_noisy_features = Feature.get_non_noisy_features(noise_threshold)
        for my_non_noisy_features in non_noisy_features:
            categories_values = list(my_non_noisy_features.categories.values())
            categories_count.append(categories_values[0].count)
            total_count = total_count + categories_values[0].count

        min_count = min(categories_count)
        percent_of_smallest_non_noisy_category = min_count/record_cnt
        smallest_sample_size = round(noise_threshold/percent_of_smallest_non_noisy_category)
        return smallest_sample_size

    def get_smallest_sample_metadata(self, noise_threshold):

        percents = [(c.name, c.count/self.total_count) for c in self.categories.values()]
        percents.sort(key = lambda c: c[1])
        smallest_category_percent = percents[0][1]
        sample_size = math.ceil(noise_threshold/smallest_category_percent)
        metadata = dict( (name, math.ceil(percent*sample_size)) for (name, percent) in percents)

        return metadata


    def count(self, category_string):
        if category_string not in self.categories:
            self.categories[category_string] = Category(category_string, self)
        category = self.categories[category_string]
        category.count += 1
        self.total_count += 1


    def get_non_noisy_categories(self, threshold=5000):
        return list(c for c in self.categories.values() if c.count >  threshold)


    def get_noisy_categories(self, threshold=5000):
        return list(c for c in self.categories.values() if c.count < threshold)

    def is_noisy(self, threshold):
        for c in self.categories.values():
            if c.count < threshold:
                return True
        return False


    def is_not_noisy(self, threshold):
        for c in self.categories.values():
            if c.count >= threshold:
                return True
        return False


    def __repr__(self):
        return f"Feature({self.name})"

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in',
                        required=True,
                        help="The data file to velocalyze.")
    parser.add_argument('--file_out',
                        required=True,
                        help="The data out file sample.")
    parser.add_argument('--feature_all',
                        required=True,
                        help="The data file to velocalyze.")
    parser.add_argument('--noise_threshold',
                        type=int,
                        default=5000,
                        help="Specifies the minimum number of (Category, Value) pairs that are required to not be considered a noisy category.")
    parser.add_argument(
        '--features',
        action='append')
    parser.add_argument(
        '--ignore',
        action='append')
    args = parser.parse_args()
    return args



def main():
    """

    """
    args = parse_command_line()
    file_in_name = args.file_in
    file_out_name = args.file_out
    feature_all = args.feature_all

    noise_threshold = args.noise_threshold
    feature_list = []
    for my_feature in args.features:
        feature_list.append(my_feature)
    ignore_list = []
    for ignore in args.ignore:
        ignore_list.append(ignore)
    data = DataSet(file_in_name)
    feature_set_non_noisy = Feature.get_non_noisy_features(noise_threshold)
    feature_set_to_sample = set()
    if feature_all == "Yes":
        feature_set_to_sample = feature_set_non_noisy
    else:
        for feat in feature_set_non_noisy:
            if feat.name in feature_list:
                feature_set_to_sample.add(feat)

    for feature in feature_set_to_sample:
        if feature.name not in ignore_list:
            print(feature.name)
            list_of_sample_for_feature = data.get_smallest_sample_feature(noise_threshold,feature)
            data.write_smallest_sample_feature_to_csv(list_of_sample_for_feature,file_out_name,feature)


if __name__ == '__main__':
    main()
