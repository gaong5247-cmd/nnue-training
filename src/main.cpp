#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace fs=std::filesystem;
constexpr uint32_t IN=768, MAXA=32; constexpr float SCALE=400.f;
#pragma pack(push,1)
struct Rec{int16_t score;int8_t result;uint8_t n;uint16_t ply;uint16_t f[MAXA];};
#pragma pack(pop)
struct DH{char magic[8];uint64_t n;uint32_t in,rs;};
struct Args{std::vector<std::string> a;std::string get(const std::string&k,const std::string&d="")const{for(size_t i=0;i+1<a.size();++i)if(a[i]==k)return a[i+1];return d;}int gi(const std::string&k,int d)const{try{return std::stoi(get(k,std::to_string(d)));}catch(...){return d;}}uint64_t gu(const std::string&k,uint64_t d)const{try{return std::stoull(get(k,std::to_string(d)));}catch(...){return d;}}float gf(const std::string&k,float d)const{try{return std::stof(get(k,std::to_string(d)));}catch(...){return d;}}};
static std::string trim(std::string s){auto p=[](unsigned char c){return !std::isspace(c);};s.erase(s.begin(),std::find_if(s.begin(),s.end(),p));s.erase(std::find_if(s.rbegin(),s.rend(),p).base(),s.end());return s;}
static int pt(char c){switch(std::tolower((unsigned char)c)){case'p':return 0;case'n':return 1;case'b':return 2;case'r':return 3;case'q':return 4;case'k':return 5;default:return -1;}}
static bool field(const std::string&s,const std::string&k,int&v){auto p=s.find(k+"=");if(p==std::string::npos)return false;p+=k.size()+1;auto e=s.find('|',p);try{v=std::stoi(trim(s.substr(p,e==std::string::npos?std::string::npos:e-p)));return true;}catch(...){return false;}}
static bool parse(const std::string&line,Rec&r){auto bar=line.find('|');if(bar==std::string::npos)return false;std::istringstream ss(trim(line.substr(0,bar)));std::string b,stm,c,e;int hm,fm;if(!(ss>>b>>stm>>c>>e>>hm>>fm)||(stm!="w"&&stm!="b"))return false;int sc=0,res=0,ply=0;if(!field(line,"score",sc))return false;field(line,"result",res);field(line,"ply",ply);r={};r.score=(int16_t)std::clamp(sc,-30000,30000);r.result=(int8_t)std::clamp(res,-1,1);r.ply=(uint16_t)std::clamp(ply,0,65535);int rank=7,file=0;bool ws=stm=="w";for(char x:b){if(x=='/'){--rank;file=0;continue;}if(x>='1'&&x<='8'){file+=x-'0';continue;}int t=pt(x);if(t<0||rank<0||file>7||r.n>=MAXA)return false;bool wp=std::isupper((unsigned char)x);int sq=rank*8+file,os=ws?sq:(sq^56),rel=(wp==ws)?0:1;r.f[r.n++]=(uint16_t)((rel*6+t)*64+os);++file;}return rank==0&&file==8;}
static void prep(const Args&a){auto ip=a.get("--input"),op=a.get("--output","train.nnueb");if(ip.empty())throw std::runtime_error("--input required");std::ifstream in(ip);if(!in)throw std::runtime_error("cannot open input");fs::create_directories(fs::path(op).parent_path().empty()?".":fs::path(op).parent_path());std::ofstream out(op,std::ios::binary|std::ios::trunc);DH h{};std::memcpy(h.magic,"NNUEB01",7);h.in=IN;h.rs=sizeof(Rec);out.write((char*)&h,sizeof(h));std::string l;uint64_t good=0,bad=0;while(std::getline(in,l)){if(trim(l).empty())continue;Rec r{};if(!parse(l,r)){++bad;continue;}out.write((char*)&r,sizeof(r));++good;}h.n=good;out.seekp(0);out.write((char*)&h,sizeof(h));std::cout<<"preprocess good="<<good<<" bad="<<bad<<"\n";if(!good)throw std::runtime_error("no valid records");}
static std::vector<Rec> load(const std::string&p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot open data");DH h{};in.read((char*)&h,sizeof(h));if(std::strncmp(h.magic,"NNUEB01",7)||h.in!=IN||h.rs!=sizeof(Rec))throw std::runtime_error("bad data");std::vector<Rec>d((size_t)h.n);in.read((char*)d.data(),(std::streamsize)(d.size()*sizeof(Rec)));if(!in)throw std::runtime_error("truncated data");return d;}
struct M{uint32_t h=64;uint64_t ep=0,seen=0;std::vector<float>w1,b1,w2;float b2=0;explicit M(uint32_t H=64):h(H),w1(IN*H),b1(H),w2(H){std::mt19937 g(12345);std::normal_distribution<float>d(0,.012f);for(auto&x:w1)x=d(g);for(auto&x:w2)x=d(g);}};
struct CH{char magic[8];uint32_t ver,in,h;uint64_t ep,seen;float b2;};
static void sv(const std::string&p,const M&m){fs::create_directories(fs::path(p).parent_path().empty()?".":fs::path(p).parent_path());std::ofstream o(p+".tmp",std::ios::binary|std::ios::trunc);CH h{};std::memcpy(h.magic,"FNNUE01",7);h.ver=1;h.in=IN;h.h=m.h;h.ep=m.ep;h.seen=m.seen;h.b2=m.b2;o.write((char*)&h,sizeof(h));o.write((char*)m.w1.data(),(std::streamsize)(m.w1.size()*4));o.write((char*)m.b1.data(),(std::streamsize)(m.b1.size()*4));o.write((char*)m.w2.data(),(std::streamsize)(m.w2.size()*4));o.close();std::error_code ec;fs::remove(p,ec);fs::rename(p+".tmp",p);}
static bool ld(const std::string&p,M&m){std::ifstream i(p,std::ios::binary);if(!i)return false;CH h{};i.read((char*)&h,sizeof(h));if(std::strncmp(h.magic,"FNNUE01",7)||h.ver!=1||h.in!=IN)throw std::runtime_error("bad checkpoint");m=M(h.h);m.ep=h.ep;m.seen=h.seen;m.b2=h.b2;i.read((char*)m.w1.data(),(std::streamsize)(m.w1.size()*4));i.read((char*)m.b1.data(),(std::streamsize)(m.b1.size()*4));i.read((char*)m.w2.data(),(std::streamsize)(m.w2.size()*4));return(bool)i;}
static float cp(const M&m,const Rec&r){std::vector<float>h(m.b1);for(uint8_t k=0;k<r.n;++k){auto*q=&m.w1[(size_t)r.f[k]*m.h];for(uint32_t j=0;j<m.h;++j)h[j]+=q[j];}float z=m.b2;for(uint32_t j=0;j<m.h;++j)z+=m.w2[j]*std::clamp(h[j],0.f,1.f);return z*SCALE;}
static float val(const M&m,const std::vector<Rec>&d){size_t n=std::min<size_t>(4096,d.size());double x=0;for(size_t i=0;i<n;++i){size_t q=(i*9973)%d.size();x+=std::abs((double)cp(m,d[q])-d[q].score);}return(float)(x/n);}
static void train(const Args&a){auto dp=a.get("--data"),ck=a.get("--checkpoint","out/checkpoint.bin"),ex=a.get("--export","out/net.nnue"),mt=a.get("--metrics","out/metrics.csv");if(dp.empty())throw std::runtime_error("--data required");int T=std::max(1,a.gi("--threads",2));
#ifdef _OPENMP
omp_set_dynamic(0);omp_set_num_threads(T);
#endif
uint32_t H=(uint32_t)std::clamp(a.gi("--hidden",64),8,256);uint64_t E=a.gu("--epochs",1000),ES=a.gu("--epoch-samples",16384),seed=a.gu("--seed",20260930);int mins=std::max(1,a.gi("--minutes",330)),save=std::max(1,a.gi("--save-every",10));float lr0=a.gf("--lr",.0025f),rw=std::clamp(a.gf("--result-weight",.1f),0.f,1.f);auto d=load(dp);M m(H);if(ld(ck,m))std::cout<<"resume epoch="<<m.ep<<" seen="<<m.seen<<"\n";else std::cout<<"new model hidden="<<H<<"\n";fs::create_directories(fs::path(mt).parent_path().empty()?".":fs::path(mt).parent_path());bool old=fs::exists(mt)&&fs::file_size(mt)>0;std::ofstream log(mt,std::ios::app);if(!old)log<<"epoch,seen,val_mae_cp,seconds,positions_per_sec\n";auto deadline=std::chrono::steady_clock::now()+std::chrono::minutes(mins);while(m.ep<E&&std::chrono::steady_clock::now()<deadline){uint64_t ep=m.ep+1,n=ES?ES:d.size();std::vector<uint64_t>ix((size_t)n);std::mt19937_64 g(seed^ep*0x9E3779B97F4A7C15ULL);if(ES){std::uniform_int_distribution<uint64_t>u(0,d.size()-1);for(auto&q:ix)q=u(g);}else{std::iota(ix.begin(),ix.end(),0ULL);std::shuffle(ix.begin(),ix.end(),g);}float lr=lr0/std::sqrt(1.f+(float)ep*.002f);auto st=std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(T)
for(long long ii=0;ii<(long long)ix.size();++ii){const Rec&r=d[(size_t)ix[(size_t)ii]];std::vector<float>h(m.h),ac(m.h),dh(m.h);for(uint32_t j=0;j<m.h;++j)h[j]=m.b1[j];for(uint8_t k=0;k<r.n;++k){float*q=&m.w1[(size_t)r.f[k]*m.h];for(uint32_t j=0;j<m.h;++j)h[j]+=q[j];}float z=m.b2;for(uint32_t j=0;j<m.h;++j){ac[j]=std::clamp(h[j],0.f,1.f);z+=m.w2[j]*ac[j];}float y=(1-rw)*std::tanh((float)r.score/SCALE)+rw*(float)r.result,p=std::tanh(z),dz=(p-y)*(1-p*p);for(uint32_t j=0;j<m.h;++j){float oldw=m.w2[j];m.w2[j]-=lr*dz*ac[j];dh[j]=dz*oldw*((h[j]>0&&h[j]<1)?1.f:0.f);m.b1[j]-=lr*dh[j];}m.b2-=lr*dz;for(uint8_t k=0;k<r.n;++k){float*q=&m.w1[(size_t)r.f[k]*m.h];for(uint32_t j=0;j<m.h;++j)q[j]-=lr*dh[j];}}
m.ep=ep;m.seen+=n;auto now=std::chrono::steady_clock::now();double s=std::chrono::duration<double>(now-st).count(),pps=s?((double)n/s):0;float mae=(ep==1||ep%10==0||ep==E)?val(m,d):-1;log<<ep<<','<<m.seen<<','<<mae<<','<<s<<','<<pps<<'\n';log.flush();std::cout<<"epoch "<<ep<<'/'<<E<<" speed="<<(uint64_t)pps<<" pos/s";if(mae>=0)std::cout<<" mae="<<mae<<"cp";std::cout<<"\n";if(ep%(uint64_t)save==0||ep==E){sv(ck,m);sv(ex,m);}}
sv(ck,m);sv(ex,m);std::cout<<"done epoch="<<m.ep<<" seen="<<m.seen<<"\n";}
static void help(){std::cout<<"nnue_train preprocess --input data.epd --output train.nnueb\n"<<"nnue_train train --data train.nnueb --epochs 1000 --epoch-samples 16384 --threads 2\n";}
int main(int argc,char**argv){try{if(argc<2){help();return 0;}Args a;for(int i=2;i<argc;++i)a.a.emplace_back(argv[i]);std::string c=argv[1];if(c=="preprocess")prep(a);else if(c=="train")train(a);else{help();return 2;}return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
