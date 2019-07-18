#pragma once

#include <storage.cuh>
#include <utils.cuh>

#include <memory>
#include <vector>

class Layer {
    public:
        Layer() {}
        // can not be = 
        // https://www.ibm.com/developerworks/cn/aix/library/1212_lufang_c11new/index.html
        Layer(const Layer *other) = delete;
        Layer(Layer &&other) = delete;
        Layer &operator=(const Layer &other) = delete;
        Layer &operator=(Layer &&other) = delete;

        // https://zhidao.baidu.com/question/1510492551209666860.html
        Layer &connect(Layer &next_layer) {
            this->next = &next_layer;
            next_layer.pre = this;

            return next_layer;
        }

        virtual void forward() { throw std::runtime_error('not implement error'); };
        virtual void backward() { throw std::runtime_error('not implement error'); };
        
        virtual std::vector<std::pair<Storage *, Storage *>> parameters() {
            throw std::runtime_error("not implement error");
        };

        virtual Storage *get_grad() { return this->grad.get(); }
        virtual Storage *get_output() { return this->get_output.get(); }

    protected:
        Layer *pre;
        Layer *next;

        std::unique_ptr<Storage> grad;
        std::unique_ptr<Storage> output;
}