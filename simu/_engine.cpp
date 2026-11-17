// Accelerated implementation of methods/*.py, checked by simu/validate.py.
// Ordinary (unsmoothed) online split conformal prediction and ACI.
// All 19 states share each replication's data and complete score history.
// Conditional disagreement masses are computed before observing the next Y.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using LD = long double;
using I64 = long long;
constexpr LD alpha = .1L, qstar = .9L;
constexpr LD pi = 3.141592653589793238462643383279502884L;

// SplitMix64: separately seeded streams make replication/chunk replay exact.
struct RNG {
    std::uint64_t state;
    explicit RNG(std::uint64_t seed) : state(seed) {}
    std::uint64_t operator()() {
        auto z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>((*this)() >> 11) * 0x1.0p-53; }
};

// Priorities affect only tree shape, never the selected order statistic.
// This internal RNG is not auxiliary randomness in the prediction algorithm.
struct History {
    struct Node { double key; std::uint64_t priority; int left=0, right=0, size=1; };
    std::vector<Node> nodes;
    int root = 0;
    RNG rng;
    History(int horizon, std::uint64_t seed) : rng(seed) {
        nodes.reserve(static_cast<std::size_t>(horizon) + 1);
        nodes.push_back({0, 0, 0, 0, 0});
    }
    int size(int i) const { return i ? nodes[i].size : 0; }
    void pull(int i) { nodes[i].size = 1 + size(nodes[i].left) + size(nodes[i].right); }
    int left(int i) {
        int j=nodes[i].right; nodes[i].right=nodes[j].left; nodes[j].left=i;
        pull(i); pull(j); return j;
    }
    int right(int i) {
        int j=nodes[i].left; nodes[i].left=nodes[j].right; nodes[j].right=i;
        pull(i); pull(j); return j;
    }
    int insert_at(int i, int j) {
        if (!i) return j;
        if (nodes[j].key < nodes[i].key) {
            nodes[i].left=insert_at(nodes[i].left,j);
            if (nodes[nodes[i].left].priority > nodes[i].priority) i=right(i);
        } else {
            nodes[i].right=insert_at(nodes[i].right,j);
            if (nodes[nodes[i].right].priority > nodes[i].priority) i=left(i);
        }
        pull(i); return i;
    }
    void insert(double score) {
        nodes.push_back({score,rng()}); root=insert_at(root,int(nodes.size())-1);
    }
    double kth(int k) const {
        int i=root;
        while (i) {
            int n=size(nodes[i].left);
            if (k==n+1) return nodes[i].key;
            if (k<=n) i=nodes[i].left;
            else { k-=n+1; i=nodes[i].right; }
        }
        throw std::runtime_error("Invalid order statistic");
    }
};

LD cdf(LD q) { return std::max(0.L, std::min(1.L,q)); }
LD threshold(const History& history, int t) {
    // Literal (1 + #{past score >= candidate}) / t > alpha, alpha=1/10.
    // Use integer division to avoid rounding the rank at t=10,20,... .
    const int k=t-t/10;
    if (k>t-1) return std::numeric_limits<LD>::infinity();
    return history.kth(k);
}
int branch(I64 errors, int t, LD bound) {
    // Form the centered candidate counts as rational numbers before comparing.
    if ((10*errors+10-t)/10.L > bound) return 1;
    if ((10*errors-t)/10.L < -bound) return -1;
    return 0;
}
struct Config {
    const char* study;
    const char* algorithm;
    int beta10;
    LD kappa;
    bool cp;
    bool motivation;
    LD beta() const { return beta10/10.L; }
};
std::vector<Config> configurations() {
    std::vector<Config> result;
    for (int b=1;b<=9;++b) result.push_back({"baseline","cp",b,.5L,true,false});
    for (int b:{3,5,7}) {
        result.push_back({"matched","cp",b,2.L,true,false});
        result.push_back({"matched","aci",b,2.L,false,false});
    }
    for (int b:{10,8,6,4}) result.push_back({"motivation","aci",b,2.L,false,true});
    return result;
}
struct State {
    LD risk=0, truncation=0, q=qstar;
    I64 realized=0, errors=0, nplus=0, nminus=0;
    LD deviation(int t) const { return (10*errors-t)/10.L; }
};
struct Options {
    int reps=500, horizon=250000, first=0;
    std::uint64_t seed=20261030;
    std::string out="simu/output/paper";
    bool trace=false;
};
Options options(int argc, char** argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        if (key=="--trace") { o.trace=true; continue; }
        if (i+1>=argc) throw std::runtime_error("Missing option value");
        std::string value=argv[++i];
        if (key=="--reps") o.reps=std::stoi(value);
        else if (key=="--horizon") o.horizon=std::stoi(value);
        else if (key=="--first-rep") o.first=std::stoi(value);
        else if (key=="--seed") o.seed=std::stoull(value);
        else if (key=="--out") o.out=value;
        else throw std::runtime_error("Unknown option: "+key);
    }
    if (o.reps<1 || o.horizon<1 || o.first<0 || o.first>std::numeric_limits<int>::max()-o.reps)
        throw std::runtime_error("Invalid dimensions");
    return o;
}
std::set<int> checkpoints(int horizon) {
    std::set<int> result={horizon};
    for (int t=100;t<=1000 && t<=horizon;t+=100) result.insert(t);
    for (int t=1200;t<=10000 && t<=horizon;t+=200) result.insert(t);
    for (int t=11000;t<=50000 && t<=horizon;t+=1000) result.insert(t);
    for (int t=60000;t<=250000 && t<=horizon;t+=10000) result.insert(t);
    return result;
}
void write_checkpoint(std::ostream& out, const Config& c, const State& s,
                      int rep, int t, LD baseline) {
    out<<c.study<<','<<c.algorithm<<','<<rep<<','<<t<<','<<c.beta()<<','<<c.kappa
       <<','<<s.risk<<','<<s.realized<<','<<s.deviation(t)<<','<<baseline
       <<','<<s.truncation<<','<<s.nplus<<','<<s.nminus<<'\n';
}

int main(int argc, char** argv) try {
    const Options o=options(argc,argv);
    const auto configs=configurations();
    std::filesystem::create_directories(o.out);
    std::ofstream records(o.out+"/checkpoints.csv"), paths(o.out+"/paths.csv"),
        diagnostics(o.out+"/diagnostics.csv"), trace;
    if (o.trace) trace.open(o.out+"/trace.csv");
    if (!records || !paths || !diagnostics || (o.trace && !trace))
        throw std::runtime_error("Cannot open output");
    for (auto* stream:{&records,&paths,&diagnostics,&trace}) *stream<<std::setprecision(20);
    records<<"study,algorithm,rep,t,beta,kappa,risk,realized,D,baseline_risk,truncation,nplus,nminus\n";
    paths<<"study,algorithm,rep,t,beta,D,coverage\n";
    diagnostics<<"rep,max_cp_violation,max_aci_violation,cp_rounds_checked,aci_rounds_checked\n";
    if (o.trace) trace<<"study,algorithm,rep,t,beta,kappa,score,baseline_threshold,baseline_p,threshold,branch,error,oracle_error,risk_increment,truncation_increment,D,history_size\n";
    std::vector<std::array<LD,10>> powers(static_cast<std::size_t>(o.horizon)+1);
    for (int t=1;t<=o.horizon;++t)
        for (int b=1;b<=10;++b) powers[t][b-1]=std::pow(LD(t),1-b/10.L);
    std::vector<bool> saved(static_cast<std::size_t>(o.horizon)+1,false);
    for (int t:checkpoints(o.horizon)) saved[t]=true;
    const auto started=std::chrono::steady_clock::now();
    for (int rep=o.first;rep<o.first+o.reps;++rep) {
        RNG split(o.seed^(std::uint64_t(rep)*0xd1b54a32d192ed03ULL));
        RNG data(split());
        History history(o.horizon,split());
        std::vector<State> states(configs.size());
        LD baseline=0, max_cp=0, max_aci=0;
        I64 cp_checked=0, aci_checked=0;
        std::vector<LD> cuts(configs.size()), losses(configs.size()), changes(configs.size());
        std::vector<int> actions(configs.size());
        for (int t=1;t<=o.horizon;++t) {
            if (history.size(history.root)!=t-1) throw std::runtime_error("Incomplete score history");
            const LD cut=threshold(history,t), p=1-cdf(cut);
            const LD baseline_increment=std::fabs(p-alpha);
            baseline+=baseline_increment;
            for (std::size_t j=0;j<configs.size();++j) {
                const auto& c=configs[j]; const auto& s=states[j];
                actions[j]=c.cp ? branch(s.errors,t,c.kappa*powers[t][c.beta10-1]) : 0;
                cuts[j]=c.cp ? (actions[j]==1 ? std::numeric_limits<LD>::infinity() :
                    actions[j]==-1 ? -std::numeric_limits<LD>::infinity() : cut) : s.q;
                losses[j]=std::fabs(cdf(cuts[j])-qstar);
                changes[j]=actions[j]==1 ? p : actions[j]==-1 ? 1-p : 0;
            }
            // DGP from simu/dgp.py; the independent validator checks every generated score.
            // Generate an actual regression observation and recompute its clipped score.
            const double X=data.uniform(), magnitude=data.uniform();
            const double sign=data.uniform()<.5 ? -1. : 1.;
            const double center=std::sin(double(2*pi*X)), scale=.2+.3*X;
            const double Y=center+scale*sign*magnitude;
            const double score=std::min(1.,std::fabs(Y-center)/scale);
            const bool oracle_error=LD(score)>qstar;
            for (std::size_t j=0;j<configs.size();++j) {
                const auto& c=configs[j]; auto& s=states[j];
                const bool error=LD(score)>cuts[j];
                s.risk+=losses[j]; s.truncation+=changes[j]; s.errors+=error;
                s.realized+=error!=oracle_error;
                s.nplus+=actions[j]==1; s.nminus+=actions[j]==-1;
                if (!c.cp) s.q+=(error-alpha)/powers[t][c.beta10-1];
                const LD violation=std::max(0.L,std::fabs(s.deviation(t))-c.kappa*powers[t][c.beta10-1]);
                if (c.cp) { max_cp=std::max(max_cp,violation); ++cp_checked; }
                else { max_aci=std::max(max_aci,violation); ++aci_checked; }
                if (violation>1e-10L) throw std::runtime_error("Every-round corridor check failed");
                if (o.trace) trace<<c.study<<','<<c.algorithm<<','<<rep<<','<<t<<','<<c.beta()<<','<<c.kappa
                    <<','<<score<<','<<cut<<','<<p<<','<<cuts[j]<<','<<actions[j]<<','<<error<<','<<oracle_error
                    <<','<<losses[j]<<','<<changes[j]<<','<<s.deviation(t)<<','<<history.size(history.root)<<'\n';
                if (saved[t]) write_checkpoint(records,c,s,rep,t,baseline);
                const bool save_baseline=c.cp && c.kappa==.5L && t<=50000 && t%100==0;
                const bool save_motivation=c.motivation && rep==0 && (t<=1000 || t%100==0);
                if (save_baseline || save_motivation) {
                    paths<<c.study<<','<<c.algorithm<<','<<rep<<','<<t<<','<<c.beta()<<','<<s.deviation(t)<<',';
                    if (save_motivation) paths<<cdf(cuts[j]);
                    paths<<'\n';
                }
            }
            // Forced decisions never discard the observation from the CP history.
            history.insert(score);
        }
        diagnostics<<rep<<','<<max_cp<<','<<max_aci<<','<<cp_checked<<','<<aci_checked<<'\n';
    }
    for (auto* stream:{&records,&paths,&diagnostics,&trace}) if (stream->is_open()) {
        stream->flush(); if (!*stream) throw std::runtime_error("Output write failed");
    }
    std::cerr<<o.reps<<" replications completed in "
        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<" s\n";
    return 0;
} catch (const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }
