import argparse
import csv

class Category(object):

    def __init__(self, name, feature):
        self.name = name
        self.feature = feature
        self.count = 0

    def is_noisy(self, threshold):
        return self.count < threshold

    def is_not_noisy(self, threshold):
        return not self.is_noisy(threshold)




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
        self.categories = {}
        Feature.features[name] = self

    def get_categories(self):
        return list(self.categories.values())

    @staticmethod
    def get_smallest_sample_size(noise_threshold):

        """

        percentage of smallest non noisy category = noise threshold / (SAMPLE SIZE I AM SEEKING)

        SAMPLE SIZE I AM SEEKING =  noise threshold / percentage of smallest non noisy category


        """


        smallest_count = []
        total_count = 0
        non_noisy_features = Feature.get_non_noisy_features(noise_threshold)
        for my_non_noisy_features in non_noisy_features:
            cat_v = list(my_non_noisy_features.categories.values())
            smallest_count.append(cat_v[0].count)
            total_count = total_count + cat_v[0].count


        min_count = min(smallest_count)
        percent_of_smallest_non_noisy_cat = min_count/total_count

        sample_size = noise_threshold/percent_of_smallest_non_noisy_cat

        return sample_size

    def get_category_percent(self, category):
        return round((category.count / self.total_count)*1000)/1000

    def count(self, category_string):
        if category_string not in self.categories:
            self.categories[category_string] = Category(category_string, self)
        category = self.categories[category_string]
        category.count += 1
        self.total_count += 1

    def get_noisy_categories(self, threshold=5000):
        return list(c for c in self.categories.values() if c.count < threshold)

    def is_noisy(self, threshold):
        for c in self.categories.values():
            if c.count < threshold:
                return True
        return False


    def get_least_not_noisy(self, threshold):

        for c in self.categories.values():

            if c.count >= threshold:

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

    parser.add_argument('--noise_threshold',
                        type=int,
                        default=5000,
                        help="Specifies the minimum number of (Category, Value) pairs that are required to not be considered a noisy category.")

    args = parser.parse_args()
    return args

def main():
    """


      """
    args = parse_command_line()
    file_in_name = args.file_in
    noise_threshold = args.noise_threshold

    with open(file_in_name, encoding='utf-8-sig') as io:
        records = list(csv.DictReader(io))

    for record in records:
        for k, v in record.items():
            feature = Feature.get_feature(k)
            feature.count(v)



    nnf = Feature.get_non_noisy_features(noise_threshold)

    s1 = Feature.get_smallest_sample_size(noise_threshold)
    print("s1 " + str(s1))

if __name__ == '__main__':
    main()
