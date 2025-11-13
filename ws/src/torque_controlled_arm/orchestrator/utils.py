import psutil

class SystemMonitor:
    @staticmethod
    def get_system_info():
        cpu_percent = psutil.cpu_percent()
        mem_info = psutil.virtual_memory()
        return {
            'cpu': cpu_percent,
            'memory': {
                'percent': mem_info.percent,
                'used': mem_info.used/1024/1024,
                'total': mem_info.total/1024/1024
            }
        }