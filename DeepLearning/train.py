import torch
import visdom
from copy import deepcopy
from tqdm import tqdm

from Lib import UtilLib
from Common import ConstVar
from DeepLearning import utils


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, device):
        """
        * 학습 관련 클래스
        :param model: 학습 시킬 모델
        :param optimizer: 학습 optimizer
        :param loss_fn: 손실 함수
        :param train_dataloader: 학습용 데이터로더
        :param device: GPU / CPU
        """

        # 학습 시킬 모델
        self.model = model
        # 학습 optimizer
        self.optimizer = optimizer
        # 손실 함수
        self.loss_fn = loss_fn
        # 학습용 데이터로더
        self.train_dataloader = train_dataloader
        # GPU / CPU
        self.device = device

    def running(self, num_epoch, output_dir, tracking_frequency, Tester, test_dataloader, metric_fn, checkpoint_file=None):
        """
        * 학습 셋팅 및 진행
        :param num_epoch: 학습 반복 횟수
        :param output_dir: 결과물 파일 저장할 디렉터리 위치
        :param tracking_frequency: 체크포인트 파일 저장 및 학습 진행 기록 빈도수
        :param Tester: 학습 성능 체크하기 위한 테스트 관련 클래스
        :param test_dataloader: 학습 성능 체크하기 위한 테스트용 데이터로더
        :param metric_fn: 학습 성능 체크하기 위한 metric
        :param checkpoint_file: 불러올 체크포인트 파일
        :return: 학습 완료 및 체크포인트 파일 생성됨
        """

        # epoch 초기화
        start_epoch_num = ConstVar.INITIAL_START_EPOCH_NUM

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.model.load_state_dict(state[ConstVar.KEY_STATE_MODEL])
            self.optimizer.load_state_dict(state[ConstVar.KEY_STATE_OPTIMIZER])
            start_epoch_num = state[ConstVar.KEY_STATE_EPOCH] + 1

        # num epoch 만큼 학습 반복
        for current_epoch_num, count in enumerate(tqdm(range(num_epoch), desc='training process'), start=start_epoch_num):

            # 학습 진행
            self._train()

            # 학습 진행 기록 주기마다 학습 진행 저장 및 시각화
            if (count + 1) % tracking_frequency == 0:

                # 현재 모델을 테스트하기 위한 테스트 객체 생성
                tester = Tester(model=deepcopy(x=self.model),
                                metric_fn=metric_fn,
                                test_dataloader=test_dataloader,
                                device=self.device)
                tester.running()

                # 체크포인트 저장
                checkpoint_dir = UtilLib.getNewPath(path=output_dir,
                                                    add=ConstVar.OUTPUT_DIR_SUFFIX_CHECKPOINT)
                checkpoint_filepath = UtilLib.getNewPath(path=checkpoint_dir,
                                                         add=ConstVar.CHECKPOINT_FILE_NAME.format(current_epoch_num))
                utils.save_checkpoint(filepath=checkpoint_filepath,
                                      model=self.model,
                                      optimizer=self.optimizer,
                                      epoch=current_epoch_num,
                                      is_best=self._check_is_best(tester=tester,
                                                                  best_checkpoint_dir=checkpoint_dir))

                # 그래프 시각화 진행
                self._draw_graph(score=tester.score,
                                 current_epoch_num=current_epoch_num,
                                 title=metric_fn.__name__)

    def _train(self):
        """
        * 학습 진행
        :return: 1 epoch 만큼 학습 진행
        """

        # 모델을 학습 모드로 전환
        self.model.train()

        # x shape: (N, 3, 32, 32)
        # y shape: (N)
        for x, y in tqdm(self.train_dataloader, desc='train dataloader', leave=False):

            # 각 텐서를 해당 디바이스로 이동
            x = x.to(self.device)
            y = y.to(self.device)

            # 순전파
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _check_is_best(self, tester, best_checkpoint_dir):
        """
        * 현재 저장하려는 모델이 가장 좋은 성능의 모델인지 여부 확인
        :param tester: 현재 모델의 성능을 테스트하기 위한 테스트 객체
        :param best_checkpoint_dir: 비교할 best 체크포인트 파일 디렉터리 위치
        :return: True / False
        """

        # best 성능 측정을 위해 초기화
        try:
            self.best_score
        except AttributeError:
            checkpoint_file = UtilLib.getNewPath(path=best_checkpoint_dir,
                                                 add=ConstVar.CHECKPOINT_BEST_FILE_NAME)
            # 기존에 측정한 best 체크포인트가 있으면 해당 score 로 초기화
            if UtilLib.isExist(checkpoint_file):
                best_tester = deepcopy(x=tester)
                best_tester.running(checkpoint_file=checkpoint_file)
                self.best_score = best_tester.score
            # 없다면 0 으로 초기화
            else:
                self.best_score = ConstVar.INITIAL_BEST_ACCURACY_ZERO

        # best 성능 갱신
        if tester.score > self.best_score:
            self.best_score = tester.score
            return True
        else:
            return False

    def _draw_graph(self, score, current_epoch_num, title):
        """
        * 학습 진행 상태 실시간으로 시각화
        :param score: 성능 평가 점수
        :param current_epoch_num: 현재 에폭 수
        :param title: 그래프 제목
        :return: visdom 으로 시각화 진행
        """

        # 서버 켜기
        try:
            self.vis
        except AttributeError:
            self.vis = visdom.Visdom()
        # 실시간으로 학습 진행 상태 그리기
        try:
            self.vis.line(Y=torch.Tensor([score]),
                          X=torch.Tensor([current_epoch_num]),
                          win=self.plt,
                          update='append',
                          opts=dict(title=title))
        except AttributeError:
            self.plt = self.vis.line(Y=torch.Tensor([score]),
                                     X=torch.Tensor([current_epoch_num]),
                                     opts=dict(title=title))
