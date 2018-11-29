import sys
import logging

import grpc
import concurrent.futures as futures

from service import common
from service.stock_prediction import StockPrediction

# Importing the generated codes from buildproto.sh
from service.service_spec import stock_prediction_pb2_grpc as grpc_bt_grpc
from service.service_spec.stock_prediction_pb2 import Output

logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("stock_prediction_service")


# Create a class to be added to the gRPC server
# derived from the protobuf codes.
class StockPredictionServicer(grpc_bt_grpc.StockPredictionServicer):
    def __init__(self):
        self.source = ""
        self.contract = ""
        self.start = ""
        self.end = ""
        self.target_date = ""

        self.output = ""

        log.info("StockPredictionServicer created")

    # The method that will be exposed to the snet-cli call command.
    # request: incoming data
    # context: object that provides RPC-specific information (timeout, etc).
    def predict(self, request, context):
        # In our case, request is a Input() object (from .proto file)
        self.source = request.source
        self.contract = request.contract
        self.start = request.start
        self.end = request.end
        self.target_date = request.target_date

        # To respond we need to create a Output() object (from .proto file)
        self.output = Output()

        sp = StockPrediction(self.source, self.contract, self.start, self.end, self.target_date)
        self.output.response = str(sp.stock_prediction()).encode("utf-8")
        log.info("stock_prediction({},{},{},{},{})={}".format(self.source,
                                                               self.contract,
                                                               self.start,
                                                               self.end,
                                                               self.target_date,
                                                               self.output.response))
        return self.output


# The gRPC serve function.
#
# Params:
# max_workers: pool of threads to execute calls asynchronously
# port: gRPC server port
#
# Add all your classes to the server here.
# (from generated .py files by protobuf compiler)
def serve(max_workers=10, port=7777):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    grpc_bt_grpc.add_StockPredictionServicer_to_server(StockPredictionServicer(), server)
    server.add_insecure_port("[::]:{}".format(port))
    return server


if __name__ == "__main__":
    """
    Runs the gRPC server to communicate with the Snet Daemon.
    """
    parser = common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    common.main_loop(serve, args)
