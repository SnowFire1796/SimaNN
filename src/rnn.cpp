#include "simann/rnn.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace simann {
namespace {
float rand_weight(std::mt19937_64& r, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    return d(r);
}
}
RNN::RNN(Config c) : H(c.hidden), cfg(c), rng(c.seed) {
    if (H < 8 || H > 512) throw std::runtime_error("hidden size must be 8..512");
    wxh.resize(VOCAB * H); whh.resize(H * H); why.resize(H * VOCAB);
    bh.assign(H, 0); by.assign(VOCAB, 0);
    const float a = std::sqrt(1.0f / H);
    for (auto& x : wxh) x = rand_weight(rng, a);
    for (auto& x : whh) x = rand_weight(rng, a);
    for (auto& x : why) x = rand_weight(rng, std::sqrt(1.0f / H));
    const std::size_t n = wxh.size()+whh.size()+why.size()+bh.size()+by.size();
    m.assign(n, 0); v.assign(n, 0);
}
std::uint64_t RNN::parameters() const {
    return wxh.size()+whh.size()+why.size()+bh.size()+by.size();
}
std::vector<float> RNN::forward(std::uint8_t token, const std::vector<float>& prev,
                                std::vector<float>& h, std::vector<float>& p) const {
    h.resize(H);
    for (int j=0;j<H;++j) {
        float z = wxh[static_cast<std::size_t>(token)*H+j] + bh[j];
        for (int k=0;k<H;++k) z += prev[k]*whh[static_cast<std::size_t>(k)*H+j];
        h[j] = std::tanh(z);
    }
    p.resize(VOCAB);
    float mx = -1e30f;
    for (int k=0;k<VOCAB;++k) {
        float z=by[k];
        for (int j=0;j<H;++j) z += h[j]*why[static_cast<std::size_t>(j)*VOCAB+k];
        p[k]=z; mx=std::max(mx,z);
    }
    float sum=0;
    for (float& x:p) { x=std::exp(x-mx); sum+=x; }
    for (float& x:p) x/=sum;
    return p;
}
float RNN::evaluate_chunk(const std::uint8_t* x, int length) const {
    if (length < 2) return 0;
    std::vector<float> prev(H,0), hidden, probs;
    double loss=0;
    for(int t=0;t<length-1;++t) {
        forward(x[t],prev,hidden,probs);
        loss -= std::log(std::max(probs[x[t+1]],1e-12f));
        prev=hidden;
    }
    return static_cast<float>(loss/(length-1));
}

float RNN::train_chunk(const std::uint8_t* x, int length) {
    if (length < 2) return 0;
    struct Cache { std::vector<float> h, p; };
    std::vector<Cache> cache(static_cast<std::size_t>(length));
    std::vector<float> prev(H,0), zero(H,0);
    float loss=0;
    for (int t=0;t<length-1;++t) {
        forward(x[t],prev,cache[t].h,cache[t].p);
        const float prob=std::max(cache[t].p[x[t+1]],1e-12f);
        loss -= std::log(prob);
        prev=cache[t].h;
    }
    const std::size_t n=wxh.size()+whh.size()+why.size()+bh.size()+by.size();
    std::vector<float> g(n,0), dhnext(H,0);
    const std::size_t o_wxh=0, o_whh=wxh.size(), o_why=o_whh+whh.size();
    const std::size_t o_bh=o_why+why.size(), o_by=o_bh+bh.size();
    for (int t=length-2;t>=0;--t) {
        auto& c=cache[t];
        std::vector<float> dy=c.p;
        dy[x[t+1]]-=1.0f;
        for (int j=0;j<H;++j)
            for (int k=0;k<VOCAB;++k)
                g[o_why+static_cast<std::size_t>(j)*VOCAB+k] += c.h[j]*dy[k];
        for (int k=0;k<VOCAB;++k) g[o_by+k]+=dy[k];
        std::vector<float> dh(H,0), dz(H,0);
        for (int j=0;j<H;++j) {
            float q=dhnext[j];
            for (int k=0;k<VOCAB;++k) q += why[static_cast<std::size_t>(j)*VOCAB+k]*dy[k];
            dh[j]=q;
            dz[j]=q*(1.0f-c.h[j]*c.h[j]);
            g[o_wxh+static_cast<std::size_t>(x[t])*H+j]+=dz[j];
            g[o_bh+j]+=dz[j];
        }
        const std::vector<float>& hp = (t==0 ? zero : cache[t-1].h);
        std::fill(dhnext.begin(),dhnext.end(),0);
        for (int i=0;i<H;++i) for (int j=0;j<H;++j) {
            g[o_whh+static_cast<std::size_t>(i)*H+j]+=hp[i]*dz[j];
            dhnext[i]+=whh[static_cast<std::size_t>(i)*H+j]*dz[j];
        }
    }
    const float inv=1.0f/static_cast<float>(length-1);
    double norm2=0;
    for (float& z:g) { z*=inv; norm2+=static_cast<double>(z)*z; }
    const float norm=static_cast<float>(std::sqrt(norm2));
    if (norm>cfg.clip_norm) { const float s=cfg.clip_norm/norm; for(float& z:g) z*=s; }
    adam_update(g);
    return loss*inv;
}
void RNN::adam_update(const std::vector<float>& g) {
    ++step;
    const float c1=1.0f-std::pow(cfg.beta1,static_cast<float>(step));
    const float c2=1.0f-std::pow(cfg.beta2,static_cast<float>(step));
    std::size_t q=0;
    auto update=[&](std::vector<float>& w) {
        for(float& z:w) {
            m[q]=cfg.beta1*m[q]+(1-cfg.beta1)*g[q];
            v[q]=cfg.beta2*v[q]+(1-cfg.beta2)*g[q]*g[q];
            const float mh=m[q]/c1, vh=v[q]/c2;
            z-=cfg.learning_rate*mh/(std::sqrt(vh)+cfg.epsilon);
            ++q;
        }
    };
    update(wxh); update(whh); update(why); update(bh); update(by);
}
std::vector<std::uint8_t> RNN::generate(const std::string& prompt,int count,float temp,int topk) {
    std::vector<float> h(H,0), nh, p;
    for(unsigned char c:prompt) { forward(c,h,nh,p); h=nh; }
    std::vector<std::uint8_t> out;
    if(prompt.empty()) { forward(static_cast<std::uint8_t>('\n'),h,nh,p); h=nh; }
    std::uniform_real_distribution<float> u(0,1);
    temp=std::max(0.05f,temp); topk=std::clamp(topk,1,VOCAB);
    for(int t=0;t<count;++t) {
        std::vector<std::pair<float,int>> order;
        order.reserve(VOCAB);
        for(int i=0;i<VOCAB;++i) order.emplace_back(p[i],i);
        std::partial_sort(order.begin(),order.begin()+topk,order.end(),
            [](auto a,auto b){return a.first>b.first;});
        float sum=0;
        for(int i=0;i<topk;++i) { order[i].first=std::pow(std::max(order[i].first,1e-12f),1.0f/temp); sum+=order[i].first; }
        float r=u(rng)*sum; int chosen=order[0].second;
        for(int i=0;i<topk;++i) { r-=order[i].first; if(r<=0){chosen=order[i].second;break;} }
        out.push_back(static_cast<std::uint8_t>(chosen));
        forward(static_cast<std::uint8_t>(chosen),h,nh,p); h=nh;
    }
    return out;
}
void RNN::save(const std::string& path) const {
    std::ofstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open checkpoint for writing");
    const char magic[8]={'S','I','M','A','R','N','N','1'};
    f.write(magic,8); f.write(reinterpret_cast<const char*>(&H),sizeof(H));
    f.write(reinterpret_cast<const char*>(&cfg),sizeof(cfg));
    f.write(reinterpret_cast<const char*>(&step),sizeof(step));
    auto put=[&](const std::vector<float>& a){std::uint64_t n=a.size();f.write(reinterpret_cast<const char*>(&n),sizeof(n));f.write(reinterpret_cast<const char*>(a.data()),static_cast<std::streamsize>(n*sizeof(float)));};
    put(wxh);put(whh);put(why);put(bh);put(by);put(m);put(v);
    if(!f) throw std::runtime_error("checkpoint write failed");
}
RNN RNN::load(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open checkpoint");
    char magic[8]; f.read(magic,8);
    if(std::string(magic,8)!="SIMARNN1") throw std::runtime_error("invalid checkpoint");
    int h; Config c; std::uint64_t st;
    f.read(reinterpret_cast<char*>(&h),sizeof(h));f.read(reinterpret_cast<char*>(&c),sizeof(c));f.read(reinterpret_cast<char*>(&st),sizeof(st));
    RNN model(c); if(model.H!=h) throw std::runtime_error("checkpoint architecture mismatch");
    auto get=[&](std::vector<float>& a){std::uint64_t n=0;f.read(reinterpret_cast<char*>(&n),sizeof(n));if(n!=a.size())throw std::runtime_error("checkpoint tensor mismatch");f.read(reinterpret_cast<char*>(a.data()),static_cast<std::streamsize>(n*sizeof(float)));};
    get(model.wxh);get(model.whh);get(model.why);get(model.bh);get(model.by);get(model.m);get(model.v);
    if(!f) throw std::runtime_error("truncated checkpoint");
    model.step=st; return model;
}
}
