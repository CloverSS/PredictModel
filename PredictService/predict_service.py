from concurrent import futures
import predict_pb2_grpc
import predict_pb2
import xgboost_predict
import arima
import grpc
import time

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class predictService(predict_pb2_grpc.PredictResourceServicer):
    # 实现 proto 文件中定义的 rpc 调用
    def Predict(self, request, context):
        print(111)
        arima_controller = arima.Arima(request)
        #xgboost_controller = xgboost_predict.Xgboost(request)
        #return predict_pb2.predictResp(result=xgboost_controller.predict())
        return predict_pb2.predictResp(result=arima_controller.auto_arima())

def serve():
    # 启动 rpc 服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    predict_pb2_grpc.add_PredictResourceServicer_to_server(predictService(), server)
    server.add_insecure_port('[::]:8092')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()