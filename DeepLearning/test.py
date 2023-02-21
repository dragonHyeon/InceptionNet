import numpy as np
from tqdm import tqdm

from Common import ConstVar
from DeepLearning import utils


class Tester:
    def __init__(self, model, metric_fn, test_dataloader, device):
        """
        * 테스트 관련 클래스
        :param model: 테스트 할 모델
        :param metric_fn: 학습 성능 체크하기 위한 metric
        :param test_dataloader: 테스트용 데이터로더
        :param device: GPU / CPU
        """

        # 테스트 할 모델
        self.model = model
        # 학습 성능 체크하기 위한 metric
        self.metric_fn = metric_fn
        # 테스트용 데이터로더
        self.test_dataloader = test_dataloader
        # GPU / CPU
        self.device = device

    def running(self, checkpoint_file=None):
        """
        * 테스트 셋팅 및 진행
        :param checkpoint_file: 불러올 체크포인트 파일
        :return: 테스트 수행됨
        """

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.model.load_state_dict(state[ConstVar.KEY_STATE_MODEL])

        # 테스트 진행
        self._test()

    def _test(self):
        """
        * 테스트 진행
        :return: score 기록
        """

        # 모델을 테스트 모드로 전환
        self.model.eval()

        # 배치 마다의 정확도 담을 리스트
        batch_accuracy_list = list()

        for x, y in tqdm(self.test_dataloader, desc='test dataloader', leave=False):

            # 각 텐서를 해당 디바이스로 이동
            x = x.to(self.device)
            y = y.to(self.device)

            # 순전파
            y_pred = self.model(x)

            # 배치 마다의 정확도 계산
            batch_accuracy_list.append(self.metric_fn(y_pred=y_pred,
                                                      y=y))

        # score 기록
        self.score = np.mean(batch_accuracy_list)
