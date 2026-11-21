#include "simann/rnn.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace simann;
static std::string arg(int argc,char**argv,const std::string& key,const std::string& def) {
    for(int i=2;i+1<argc;++i) if(argv[i]==key) return argv[i+1];
    return def;
}
static int num(const std::string&s,int d){try{return std::stoi(s);}catch(...){return d;}}
static float real(const std::string&s,float d){try{return std::stof(s);}catch(...){return d;}}
static void usage(){
 std::cout<<"SimaNN - CPU text-generation RNN (untrained by default)\n"
 <<"Commands:\n  simann init --model model.snn [--hidden 96]\n"
 <<"  simann train --data corpus.txt --model model.snn [--steps 10000] [--seq 64]\n"
 <<"  simann generate --model model.snn [--prompt \"سلام\"] [--tokens 300]\n"
 <<"  simann info --model model.snn\n";
}
int main(int argc,char**argv){
 try {
  if(argc<2){usage();return 0;}
  std::string cmd=argv[1], modelPath=arg(argc,argv,"--model","sima.snn");
  if(cmd=="init"){
   Config c;c.hidden=num(arg(argc,argv,"--hidden","96"),96);
   RNN model(c);model.save(modelPath);
   std::cout<<"Initialized random, untrained model: "<<modelPath<<"\nParameters: "<<model.parameters()<<"\n";
   return 0;
  }
  if(cmd=="info"){
   RNN m=RNN::load(modelPath);
   std::cout<<"Architecture: byte-level Elman RNN\nHidden units: "<<m.hidden_size()<<"\nParameters: "<<m.parameters()<<"\nCheckpoint: "<<modelPath<<"\n";
   return 0;
  }
  if(cmd=="generate"){
   RNN m=RNN::load(modelPath);
   std::string prompt=arg(argc,argv,"--prompt","\n");
   int count=num(arg(argc,argv,"--tokens","300"),300);
   float temp=real(arg(argc,argv,"--temperature","0.85"),0.85f);
   int topk=num(arg(argc,argv,"--top-k","32"),32);
   auto bytes=m.generate(prompt,count,temp,topk);
   std::cout<<prompt;
   std::cout.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
   std::cout<<"\n"; return 0;
  }
  if(cmd=="train"){
   std::string data=arg(argc,argv,"--data","");
   if(data.empty()) throw std::runtime_error("--data is required");
   std::ifstream in(data,std::ios::binary|std::ios::ate);
   if(!in) throw std::runtime_error("cannot open dataset");
   const auto size=in.tellg();
   if(size<4) throw std::runtime_error("dataset must contain at least four bytes for a train/validation split");
   const std::uint64_t total=static_cast<std::uint64_t>(size);
   const std::uint64_t split=std::clamp<std::uint64_t>(total*9/10,2,total-2);
   const std::uint64_t validationBytes=total-split;
   std::ifstream val(data,std::ios::binary);
   Config c;c.hidden=num(arg(argc,argv,"--hidden","96"),96);
   c.sequence=num(arg(argc,argv,"--seq","64"),64);
   c.learning_rate=real(arg(argc,argv,"--lr","0.001"),0.001f);
   const int steps=num(arg(argc,argv,"--steps","10000"),10000);
   RNN m;
   try { m=RNN::load(modelPath); std::cout<<"Resuming checkpoint "<<modelPath<<"\n"; }
   catch(...) { m=RNN(c); std::cout<<"Starting a new randomly initialized model\n"; }
   std::vector<std::uint8_t> buf(static_cast<std::size_t>(c.sequence+1));
   in.clear();in.seekg(0);
   double sum=0; auto start=std::chrono::steady_clock::now();
   for(int s=1;s<=steps;++s){
    auto pos=in.tellg();
    if(pos<0 || static_cast<std::uint64_t>(pos)+2>=split){in.clear();in.seekg(0);pos=0;}
    const auto available=split-static_cast<std::uint64_t>(pos);
    const auto want=std::min<std::uint64_t>(buf.size(),available);
    in.read(reinterpret_cast<char*>(buf.data()),static_cast<std::streamsize>(want));
    auto got=static_cast<int>(in.gcount());
    if(got<2){in.clear();in.seekg(0);in.read(reinterpret_cast<char*>(buf.data()),static_cast<std::streamsize>(std::min<std::uint64_t>(buf.size(),split)));got=static_cast<int>(in.gcount());}
    if(got<2) throw std::runtime_error("not enough training bytes");
    sum+=m.train_chunk(buf.data(),got);
    if(s%100==0 || s==1){
     const auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
     std::cout<<"step "<<s<<"/"<<steps<<" loss "<<(sum/std::min(s,100))<<" elapsed "<<sec<<"s\n";
     sum=0;
    }
    if(s%1000==0) m.save(modelPath);
   }
   m.save(modelPath);
   std::cout<<"Training complete. Saved "<<modelPath<<"\n";
   std::cout<<"Note: validation split reserved at end of corpus; this compact trainer currently reports training loss only.\n";
   return 0;
  }
  usage();return 1;
 } catch(const std::exception&e){std::cerr<<"SimaNN error: "<<e.what()<<"\n";return 2;}
}
