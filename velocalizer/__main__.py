import argparse
import csv
import math
import random
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
        print("TOTAL RECORD COUNT " + str(record_cnt))

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

    parser.add_argument('--noise_threshold',
                        type=int,
                        default=5000,
                        help="Specifies the minimum number of (Category, Value) pairs that are required to not be considered a noisy category.")

    args = parser.parse_args()
    return args


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


    def get_smallest_sample_feature(self, noise_threshold,non_noisy_feature):
        random.shuffle(self.records)
        new_records = []
        category_count = {}
        metadata = non_noisy_feature.get_smallest_sample_metadata(noise_threshold)
        for category in metadata:
            category_count[category] = metadata[category]
        for r in self.records:
            target_category = str(r[non_noisy_feature.name])
            if category_count[target_category] > 0:
                new_records.append(r)
                category_count[target_category] = category_count[target_category] - 1
                if category_count[target_category] == 0:
                    return new_records

        raise Exception("Could not build smallest sample")



        return new_records



    def get_non_noisy_features(self, noise_threshold,outfile):
        return Feature.get_non_noisy_features(noise_threshold)



    def write_smallest_non_noisy_sample(self, noise_threshold,outfile):


        random.shuffle(self.records)

        curr_count1 = 0
        curr_count2 = 0
        non_noisy_features = Feature.get_non_noisy_features(noise_threshold)

        for record in self.records:
            print(record)
            for non_noisy_feature in non_noisy_features:
                meta = non_noisy_feature.get_smallest_sample_metadata(noise_threshold)
                print("meta VALUES " + str(meta))
                print("feature.name KEYS " + str(non_noisy_feature.name))
                print("record[non_noisy_feature.name] " + str(record[non_noisy_feature.name]))

                ss = self.get_smallest_sample(noise_threshold,non_noisy_feature.name)

                print("ss VALUES " + str(ss))








        return

def main():
    """

    """
    args = parse_command_line()
    file_in_name = args.file_in
    file_out_name = args.file_out

    noise_threshold = args.noise_threshold
    data = DataSet(file_in_name)
    #data.write_smallest_non_noisy_sample(noise_threshold,file_out_name)


    non_noisy_features = Feature.get_non_noisy_features(noise_threshold)

    for feat in non_noisy_features:
        new_recs = data.get_smallest_sample_feature(noise_threshold,feat)
        print(new_recs)
        exit()



if __name__ == '__main__':
    main()
