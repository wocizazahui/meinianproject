from constants import MEINIAN_DATASET_PART2, MEINIAN_DATASET_PART1, TEST_FILE_PATH, TRAINING_FILE_PATH

ID = 0
TEST_ID = 1
TEST_RESULT = 2


class MeinianData(object):

    def __init__(self, training_set_path, test_set_path, dataset_path):

        self.dataset_path = dataset_path
        self.training_set_path = training_set_path
        self.test_set_path = test_set_path

        self.dataset = {}
        self.set_up_data_set()

        self.training_set = {}
        self.set_up_training_set()

        self.test_set = []
        self.set_up_test_set()
        self.ZAZAHUI={}

    def set_up_data_set(self):
        for file_path in self.dataset_path:
            with open(file_path, 'r', encoding='utf-8') as data_file:
                next(data_file)
                for line in data_file.readlines():
                    #print(line)
                    data = line.replace('\n', '').split("$")
                    #print(data)
                    if data[ID] in self.dataset:
                        self.dataset[data[ID]][data[TEST_ID]] = data[TEST_RESULT]
                    else:
                        self.dataset[data[ID]] = {data[TEST_ID]: data[TEST_RESULT]}


    def set_up_training_set(self):
        with open(self.training_set_path, 'r') as training_file:
            next(training_file)
            for line in training_file.readlines():
                data = line.replace('\n', '').split(',', maxsplit=1)
                self.training_set[data[ID]] = data[ID+1].split(',')
                #print(data)

    def set_up_test_set(self):
        with open(self.training_set_path, 'r') as test_file:
            next(test_file)
            for line in test_file.readlines():
                data = line.replace('\n', '').split(',', maxsplit=1)
                self.test_set.append(data[ID])
                #print(data)


def get_test_report_by_test_id(meinian_data, test_id):

    result = []
    for person_id in meinian_data.dataset.keys():
        if test_id in meinian_data.dataset[person_id].keys():
            result.append(meinian_data.dataset[person_id][test_id])

    return result


def get_all_test_id(meinian_data):
    test_ids = set()
    for key in meinian.dataset.keys():
        person = meinian.dataset[key]
        test_ids = test_ids.union(set(meinian.dataset[key].keys()))

    test_ids = list(test_ids)
    test_ids.sort()

    return test_ids


if __name__ == "__main__":
    meinian = MeinianData(TRAINING_FILE_PATH,
                          TEST_FILE_PATH,
                          [MEINIAN_DATASET_PART1, MEINIAN_DATASET_PART2])

    # get test result by test id
    result = get_test_report_by_test_id(meinian, '0113')

    # get all tests
    result = get_all_test_id(meinian)
    
