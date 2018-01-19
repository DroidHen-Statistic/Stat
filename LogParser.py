import abc
class LogParser(abc.ABC):

    @abc.abstractmethod
    def parse(self):
        """
        log解析
        """

    @abc.abstractmethod
    def output_to_file(self):
        """
        结果输出
        """