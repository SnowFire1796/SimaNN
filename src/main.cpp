#include "simann/rnn.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace simann;
static std::string arg(int argc,char** argv,const std::string& key,const std::string& def) {
    for(int i=2;i+1<argc;++i) if(std::string(argv[i])==key) return argv[i+1];
    return def;
}
static int num(const std::string& s,int d) { try { return std::stoi(s); } catch(...) { return d; } }
static float real(const std::string& s,float d) { try { return std::stof(s); } catch(...) { return d; } }
static void usage() {
    std::cout << "SimaNN - CPU byte-level text-generation RNN (untrained by default)\n"
              << "Commands:\n"
              << "  simann init --model model.snn [--hidden 96]\n"
              << "  simann train --data corpus.txt --model model.snn [--steps 10000] [--seq 64]\n"
              << "  simann generate --model model.snn [--prompt \"سلام\"] [--tokens 300]\n"
              << "  simann info --model model.snn\n";
}
int main(int argc,char** argv) {
 try {
    if(argc<2) { usage(); return 0; }
    const std::string cmd=argv[1];
    const std::string modelPath=arg(argc,argv,"--model","sima.snn");
    if(cmd=="init") {
        Config c; c.hidden=num(arg(argc,argv,"--hidden","96"),96);
        RNN model(c); model.save(modelPath);
        std::cout<<"Initialized random, untrained model: "<<modelPath
                 <<"\nParameters: "<<model.parameters()<<"\n";
        return 0;
    }
    if(cmd=="info") {
        const RNN model=RNN::load(modelPath);
        std::cout<<"Architecture: byte-level Elman RNN\nHidden units: "<<model.hidden_size()
                 <<"\nParameters: "<<model.parameters()<<"\nCheckpoint: "<<modelPath<<"\n";
        return 0;
    }
    if(cmd=="generate") {
        RNN model=RNN::load(modelPath);
        const std::string prompt=arg(argc,argv,"--prompt","\n");
        const int count=std::max(0,num(arg(argc,argv,"--tokens","300"),300));
        const float temp=real(arg(argc,argv,"--temperature","0.85"),0.85f);
        const int topk=num(arg(argc,argv,"--top-k","32"),32);
        const auto bytes=model.generate(prompt,count,temp,topk);
        std::cout<<prompt;
        std::cout.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
        std::cout<<"\n";
        return 0;
    }
    if(cmd=="train") {
        const std::string data=arg(argc,argv,"--data","");
        if(data.empty()) throw std::runtime_error("--data is required");
        std::ifstream in(data,std::ios::binary|std::ios::ate);
        if(!in) throw std::runtime_error("cannot open dataset");
        const auto end=in.tellg();
        if(end<4) throw std::runtime_error("dataset must contain at least four bytes");
        const std::uint64_t total=static_cast<std::uint64_t>(end);
        const std::uint64_t split=std::clamp<std::uint64_t>(total*9/10,2,total-2);
        const std::uint64_t validationBytes=total-split;
        std::ifstream val(data,std::ios::binary);
        if(!val) throw std::runtime_error("cannot open validation stream");

        Config c;
        c.hidden=num(arg(argc,argv,"--hidden","96"),96);
        c.sequence=std::clamp(num(arg(argc,argv,"--seq","64"),64),2,4096);
        c.learning_rate=real(arg(argc,argv,"--lr","0.001"),0.001f);
        const int steps=std::max(1,num(arg(argc,argv,"--steps","10000"),10000));
        RNN model;
        try {
            model=RNN::load(modelPath);
            std::cout<<"Resuming checkpoint "<<modelPath<<"\n";
        } catch(...) {
            model=RNN(c);
            std::cout<<"Starting a new randomly initialized model\n";
        }

        std::vector<std::uint8_t> buffer(static_cast<std::size_t>(c.sequence+1));
        in.clear(); in.seekg(0);
        double runningLoss=0.0;
        auto started=std::chrono::steady_clock::now();
        for(int step=1;step<=steps;++step) {
            auto pos=in.tellg();
            if(pos<0 || static_cast<std::uint64_t>(pos)+2>=split) {
                in.clear(); in.seekg(0); pos=0;
            }
            const std::uint64_t available=split-static_cast<std::uint64_t>(pos);
            const std::uint64_t wanted=std::min<std::uint64_t>(buffer.size(),available);
            in.read(reinterpret_cast<char*>(buffer.data()),static_cast<std::streamsize>(wanted));
            int got=static_cast<int>(in.gcount());
            if(got<2) {
                in.clear(); in.seekg(0);
                in.read(reinterpret_cast<char*>(buffer.data()),static_cast<std::streamsize>(std::min<std::uint64_t>(buffer.size(),split)));
                got=static_cast<int>(in.gcount());
            }
            if(got<2) throw std::runtime_error("training partition is too small");
            runningLoss+=model.train_chunk(buffer.data(),got);

            if(step%100==0 || step==1 || step==steps) {
                double validationLoss=0.0;
                int validationWindows=0;
                for(int k=0;k<8;++k) {
                    const std::uint64_t offset=(static_cast<std::uint64_t>(k)*buffer.size())%validationBytes;
                    if(validationBytes-offset<2) continue;
                    val.clear();
                    val.seekg(static_cast<std::streamoff>(split+offset));
                    const auto count=std::min<std::uint64_t>(buffer.size(),validationBytes-offset);
                    val.read(reinterpret_cast<char*>(buffer.data()),static_cast<std::streamsize>(count));
                    const int n=static_cast<int>(val.gcount());
                    if(n>=2) { validationLoss+=model.evaluate_chunk(buffer.data(),n); ++validationWindows; }
                }
                const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                const int window=step%100==0?100:(step%100);
                std::cout<<"step "<<step<<"/"<<steps
                         <<" train_loss "<<(runningLoss/window)
                         <<" val_loss "<<(validationWindows?validationLoss/validationWindows:0.0)
                         <<" elapsed "<<elapsed<<"s\n";
                runningLoss=0.0;
            }
            if(step%1000==0) model.save(modelPath);
        }
        model.save(modelPath);
        std::cout<<"Training complete. Checkpoint saved to "<<modelPath
                 <<" (held-out final 10% used for validation).\n";
        return 0;
    }
    usage();
    return 1;
 } catch(const std::exception& e) {
    std::cerr<<"SimaNN error: "<<e.what()<<"\n";
    return 2;
 }
}
