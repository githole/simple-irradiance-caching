// Copyright (c) 2014 hole
// This software is released under the MIT License (http://kagamin.net/hole/license.txt).
// A part of this software is based on smallpt (http://www.kevinbeason.com/smallpt/) and
// released under the MIT License (http://kagamin.net/hole/smallpt-license.txt).
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <queue>

const double PI = 3.14159265358979323846;
const double INF = 1e20;
const double EPS = 1e-6;
const double MaxDepth = 5;

// *** その他の関数 ***
inline double rand01() { return (double)rand()/RAND_MAX; }

// *** データ構造 ***
struct Vec {
	double x, y, z;
	Vec(const double x_ = 0, const double y_ = 0, const double z_ = 0) : x(x_), y(y_), z(z_) {}
	inline Vec operator+(const Vec &b) const {return Vec(x + b.x, y + b.y, z + b.z);}
	inline Vec operator-(const Vec &b) const {return Vec(x - b.x, y - b.y, z - b.z);}
	inline Vec operator*(const double b) const {return Vec(x * b, y * b, z * b);}
	inline Vec operator/(const double b) const {return Vec(x / b, y / b, z / b);}
	inline const double LengthSquared() const { return x*x + y*y + z*z; }
	inline const double Length() const { return sqrt(LengthSquared()); }
};
inline Vec operator*(double f, const Vec &v) { return v * f; }
inline Vec Normalize(const Vec &v) { return v / v.Length(); }
// 要素ごとの積をとる
inline const Vec Multiply(const Vec &v1, const Vec &v2) {
	return Vec(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
inline const double Dot(const Vec &v1, const Vec &v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline const Vec Cross(const Vec &v1, const Vec &v2) {
	return Vec((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x));
}
typedef Vec Color;
const Color BackgroundColor(0.0, 0.0, 0.0);

struct Ray {
	Vec org, dir;
	Ray(const Vec org_, const Vec &dir_) : org(org_), dir(dir_) {}
};

enum ReflectionType {
	DIFFUSE,    // 完全拡散面。いわゆるLambertian面。
};

struct Sphere {
	double radius;
	Vec position;
	Color emission, color;
	ReflectionType ref_type;

	Sphere(const double radius_, const Vec &position_, const Color &emission_, const Color &color_, const ReflectionType ref_type_) :
	  radius(radius_), position(position_), emission(emission_), color(color_), ref_type(ref_type_) {}
	// 入力のrayに対する交差点までの距離を返す。交差しなかったら0を返す。
	const double intersect(const Ray &ray) {
		Vec o_p = position - ray.org;
		const double b = Dot(o_p, ray.dir), det = b * b - Dot(o_p, o_p) + radius * radius;
		if (det >= 0.0) {
			const double sqrt_det = sqrt(det);
			const double t1 = b - sqrt_det, t2 = b + sqrt_det;
			if (t1 > EPS)		return t1;
			else if(t2 > EPS)	return t2;
		}
		return 0.0;
	}
};


// 以下フォトンマップ用データ構造
struct Photon {
	Vec position;
	Color power;
	Vec incident;

	Photon(const Vec& position_, const Color& power_, const Vec& incident_) :
	position(position_), power(power_), incident(incident_) {}
};

// KD-tree
template<typename T>
class KDTree {
public:
	// k-NN searchのクエリ
	struct Query {
		double max_distance2; // 探索の最大半径
		size_t max_search_num; // 最大探索点数
		Vec search_position; // 探索中心
		Vec normal; // 探索中心における法線
		Query(const Vec &search_position_, const Vec &normal_, const double max_distance2_, const size_t max_search_num_) :
		max_distance2(max_distance2_), normal(normal_), max_search_num(max_search_num_), search_position(search_position_) {}
	};
	// 結果のQueueに乗せるためのデータ構造。
	struct ElementForQueue {
		const T *point;
		double distance2;
		ElementForQueue(const T *point_, const double distance2_) : point(point_), distance2(distance2_) {}
		bool operator<(const ElementForQueue &b) const {
			return distance2 < b.distance2;
		}
	};
	// KNNの結果を格納するキュー
	typedef std::priority_queue<ElementForQueue, std::vector<ElementForQueue> > ResultQueue;
protected:
	std::vector<T> points;
	struct KDTreeNode {
		T* point;
		KDTreeNode* left;
		KDTreeNode* right;
		int axis;

		KDTreeNode() : point(NULL), left(NULL), right(NULL) {
		}
	};
	KDTreeNode* root;
	void delete_kdtree(KDTreeNode* node) {
		if (node == NULL)
			return;
		delete_kdtree(node->left);
		delete_kdtree(node->right);
		delete node;
	}

	// フツーのk-NN search。
	void locate_points(typename KDTree<T>::ResultQueue* pqueue, KDTreeNode* node, typename KDTree<T>::Query &query) {
		if (node == NULL)
			return;
		const int axis = node->axis;

		double delta;
		switch (axis) {
		case 0: delta = query.search_position.x - node->point->position.x; break;
		case 1: delta = query.search_position.y - node->point->position.y; break;
		case 2: delta = query.search_position.z - node->point->position.z; break;
		}

		// 対象点<->探索中心の距離が設定半径以下　かつ　対象点<->探索中心の法線方向の距離が一定以下　という条件ならその対象点格納
		const Vec dir = node->point->position - query.search_position;
		const double distance2 = dir.LengthSquared();
		const double dt = Dot(query.normal, dir / sqrt(distance2));
		if (distance2 < query.max_distance2 && fabs(dt) <= query.max_distance2 * 0.01) {
			pqueue->push(ElementForQueue(node->point, distance2));
			if (pqueue->size() > query.max_search_num) {
				pqueue->pop();
				query.max_distance2 = pqueue->top().distance2;
			}
		}
		if (delta > 0.0) { // みぎ
			locate_points(pqueue,node->right, query);
			if (delta * delta < query.max_distance2) {
				locate_points(pqueue, node->left, query);
			}
		} else { // ひだり
			locate_points(pqueue,node->left, query);
			if (delta * delta < query.max_distance2) {
				locate_points(pqueue, node->right, query);
			}
		}

	}
	
	static bool kdtree_less_operator_x(const T& left, const T& right) {
		return left.position.x < right.position.x;
	}
	static bool kdtree_less_operator_y(const T& left, const T& right) {
		return left.position.y < right.position.y;
	}
	static bool kdtree_less_operator_z(const T& left, const T& right) {
		return left.position.z < right.position.z;
	}
	
	KDTreeNode* create_kdtree_sub(typename std::vector<T>::iterator begin,typename std::vector<T>::iterator end, int depth) {
		if (end - begin <= 0) {
			return NULL;
		}
		const int axis = depth % 3;
		// 中央値
		switch (axis) {
		case 0: std::sort(begin, end, kdtree_less_operator_x); break;
		case 1: std::sort(begin, end, kdtree_less_operator_y); break;
		case 2: std::sort(begin, end, kdtree_less_operator_z); break;
		}
		const int median = (end - begin) / 2;
		KDTreeNode* node = new KDTreeNode;
		node->axis = axis;
		node->point = &(*(begin + median));
		// 子供
		node->left  = create_kdtree_sub(begin,              begin + median, depth + 1);
		node->right = create_kdtree_sub(begin + median + 1, end,            depth + 1);
		return node;
	}

public:
	KDTree() {
		root = NULL;
	}
	virtual ~KDTree() {
		delete_kdtree(root);
	}
	size_t Size() {
		return points.size();
	}
	void SearchKNN(typename KDTree::ResultQueue* pqueue, typename KDTree<T>::Query &query) {
		locate_points(pqueue, root, query);
	}
	void AddPoint(const T &point) {
		points.push_back(point);
	}
	void CreateKDtree() {
		root = create_kdtree_sub(points.begin(), points.end(), 0);
	}
};
typedef KDTree<Photon> PhotonMap;

// イラディアンスキャッシュ
struct Irradiance {
	Vec position;
	Color irradiance;
	Vec normal;
	double R0;

	Irradiance(const Vec& position_, const Color& irradiance_, const Vec& normal_, const double R0_) :
	irradiance(irradiance_), position(position_), normal(normal_), R0(R0_) {
	}
};

class IrradianceCache : public KDTree<Irradiance> {
public:
	
	// k-NN searchのクエリ
	struct IrradianceQuery {
		const Vec search_position; // 探索中心
		const Vec normal; // 探索中心における法線
		const double threashold;
		const double max_distance2;

		IrradianceQuery(const Vec &search_position_, const Vec &normal_, const double max_distance2_, const double threashold_) :
		normal(normal_), max_distance2(max_distance2_), search_position(search_position_), threashold(threashold_) {}
	};
	// 結果のQueueに乗せるためのデータ構造。
	struct ElementForIrradianceQueue {
		const Irradiance *point;
		double distance2;
		double weight;

		ElementForIrradianceQueue(const Irradiance *point_, const double weight_) : point(point_), weight(weight_) {}
		bool operator<(const ElementForIrradianceQueue &b) const {
			return distance2 < b.distance2;
		}
		
		ElementForIrradianceQueue& operator=(const ElementForIrradianceQueue& obj) {
			this->point = obj.point;
			this->distance2 = obj.distance2;
			this->weight = obj.weight;
			return (*this);
		}
	};
	// KNNの結果を格納するキュー
	typedef std::priority_queue<ElementForIrradianceQueue, std::vector<ElementForIrradianceQueue> > ResultIrradianceQueue;

private:
	std::vector<Irradiance*> ptrs;

	// フツーのk-NN search。
	void locate_irradiance_points(IrradianceCache::ResultIrradianceQueue* pqueue, KDTreeNode* node, IrradianceQuery &query) {
		if (node == NULL)
			return;
		const int axis = node->axis;

		double delta;
		switch (axis) {
		case 0: delta = query.search_position.x - node->point->position.x; break;
		case 1: delta = query.search_position.y - node->point->position.y; break;
		case 2: delta = query.search_position.z - node->point->position.z; break;
		}

		const Vec dir = node->point->position - query.search_position;
		const double distance2 = dir.LengthSquared();
		const double dt = Dot(query.normal, dir / sqrt(distance2));

		// イラディアンスキャッシュの重み
		const double weight = 1.0 / ((query.search_position - node->point->position).Length() / node->point->R0 + sqrt(1.0 - Dot(query.normal, node->point->normal)));
		if (weight > query.threashold) {
			pqueue->push(ElementForIrradianceQueue(node->point, weight));
		}
		
		if (delta > 0.0) { // みぎ
			locate_irradiance_points(pqueue, node->right, query);
			if (delta * delta < query.max_distance2) {
				locate_irradiance_points(pqueue, node->left, query);
			}
		} else { // ひだり
			locate_irradiance_points(pqueue,node->left, query);
			if (delta * delta < query.max_distance2) {
				locate_irradiance_points(pqueue, node->right, query);
			}
		}
	}
	
	void add_new_point(KDTreeNode* node, const Irradiance& point) {
		const int axis = node->axis;
		double delta;
		switch (axis) {
		case 0: delta = point.position.x - node->point->position.x; break;
		case 1: delta = point.position.y - node->point->position.y; break;
		case 2: delta = point.position.z - node->point->position.z; break;
		}

		if (delta > 0.0) { // みぎ
			if (node->right != NULL) {
				add_new_point(node->right, point);
			} else {
				KDTreeNode* newnode = new KDTreeNode;
				newnode->axis = (axis + 1) % 3;
				newnode->point = new Irradiance(point);
				ptrs.push_back(newnode->point);
				node->right = newnode;
			}
		} else { // ひだり
			if (node->left != NULL) {
				add_new_point(node->left, point);
			} else {
				KDTreeNode* newnode = new KDTreeNode;
				newnode->axis = (axis + 1) % 3;
				newnode->point = new Irradiance(point);
				ptrs.push_back(newnode->point);
				node->left = newnode;
			}
		}
	}
public:
	~IrradianceCache() {	
		for (int i = 0; i < ptrs.size(); i ++)
			delete ptrs[i];
	}
	void SearchCachedPoints(ResultIrradianceQueue* iqueue, IrradianceQuery &query) {
		locate_irradiance_points(iqueue, root, query);
	}
	void AddPointToTree(const Irradiance &point) {
		if (root != NULL) {
			add_new_point(root, point);
		} else {
			KDTreeNode* newnode = new KDTreeNode;
			newnode->axis = 0;
			newnode->point = new Irradiance(point);
			ptrs.push_back(newnode->point);
			root = newnode;
		}
	}
};


// *** レンダリングするシーンデータ ****
// from smallpt
Sphere spheres[] = {
	Sphere(5.0, Vec(50.0, 75.0, 81.6),Color(12,12,12), Color(), DIFFUSE),//照明
	Sphere(1e5, Vec( 1e5+1,40.8,81.6), Color(), Color(0.75, 0.25, 0.25),DIFFUSE),// 左
	Sphere(1e5, Vec(-1e5+99,40.8,81.6),Color(), Color(0.25, 0.25, 0.75),DIFFUSE),// 右
	Sphere(1e5, Vec(50,40.8, 1e5),     Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 奥
	Sphere(1e5, Vec(50,40.8,-1e5+170), Color(), Color(), DIFFUSE),// 手前
	Sphere(1e5, Vec(50, 1e5, 81.6),    Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 床
	Sphere(1e5, Vec(50,-1e5+81.6,81.6),Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 天井
	Sphere(16.5,Vec(27,16.5,47),       Color(), Color(0.75, 0.75, 0.75), DIFFUSE),// 鏡
	Sphere(16.5,Vec(73,16.5,78),       Color(), Color(0.75, 0.75, 0.75), DIFFUSE),//ガラス
};
const int LightID = 0;

// *** レンダリング用関数 ***
// シーンとの交差判定関数
inline bool intersect_scene(const Ray &ray, double *t, int *id) {
	const double n = sizeof(spheres) / sizeof(Sphere);
	*t  = INF;
	*id = -1;
	for (int i = 0; i < int(n); i ++) {
		double d = spheres[i].intersect(ray);
		if (d > 0.0 && d < *t) {
			*t  = d;
			*id = i;
		}
	}
	return *t < INF;
}

// フォトン追跡法によりフォトンマップ構築
void create_photon_map(const int shoot_photon_num, PhotonMap *photon_map) {
	std::cout << "Shooting photons... (" << shoot_photon_num << " photons)" << std::endl;
	for (int i = 0; i < shoot_photon_num; i ++) {
		// 光源からフォトンを発射する
		// 光源上の一点をサンプリングする	
		const double r1 = 2 * PI * rand01();
		const double r2 = 1.0 - 2.0 * rand01() ;
		const Vec light_pos = spheres[LightID].position + ((spheres[LightID].radius + EPS) * Vec(sqrt(1.0 - r2*r2) * cos(r1), sqrt(1.0 - r2*r2) * sin(r1), r2));

		const Vec normal = Normalize(light_pos - spheres[LightID].position);
		// 光源上の点から半球サンプリングする
		Vec w, u, v;
		w = normal;
		if (fabs(w.x) > 0.1)
			u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
		else
			u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
		v = Cross(w, u);
		// コサイン項に比例させる。フォトンが運ぶのが放射輝度ではなく放射束であるため。
		const double u1 = 2 * PI * rand01();
		const double u2 = rand01(), u2s = sqrt(u2);
		Vec light_dir = Normalize((u * cos(u1) * u2s + v * sin(u1) * u2s + w * sqrt(1.0 - u2)));

		Ray now_ray(light_pos, light_dir);
		// emissionの値は放射輝度だが、フォトンが運ぶのは放射束なので変換する必要がある。
		// L（放射輝度）= dΦ/(cosθdωdA)なので、光源の放射束はΦ = ∫∫L・cosθdωdAになる。今回は球光源で完全拡散光源であることから
		// 球上の任意の場所、任意の方向に等しい放射輝度Leを持つ。（これがemissionの値）よって、
		// Φ = Le・∫∫cosθdωdAで、Le・∫dA∫cosθdωとなり、∫dAは球の面積なので4πr^2、∫cosθdωは立体角の積分なのでπとなる。
		// よって、Φ = Le・4πr^2・πとなる。この値を光源から発射するフォトン数で割ってやれば一つのフォトンが運ぶ放射束が求まる。
		Color now_flux = spheres[LightID].emission * 4.0 * PI * pow(spheres[LightID].radius, 2.0) * PI / shoot_photon_num;

		// フォトンがシーンを飛ぶ
		bool trace_end = false;
		for (;!trace_end;) {
			// 放射束が0.0なフォトンを追跡してもしょうがないので打ち切る
			if (std::max(now_flux.x, std::max(now_flux.y, now_flux.z)) <= 0.0)
				break;

			double t; // レイからシーンの交差位置までの距離
			int id;   // 交差したシーン内オブジェクトのID
			if (!intersect_scene(now_ray, &t, &id))
				break;
			const Sphere &obj = spheres[id];
			const Vec hitpoint = now_ray.org + t * now_ray.dir; // 交差位置
			const Vec normal  = Normalize(hitpoint - obj.position); // 交差位置の法線
			const Vec orienting_normal = Dot(normal, now_ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）

			switch (obj.ref_type) {
			case DIFFUSE: {
				// 拡散面なのでフォトンをフォトンマップに格納する
				photon_map->AddPoint(Photon(hitpoint, now_flux, now_ray.dir));

				// 反射するかどうかをロシアンルーレットで決める
				// 例によって確率は任意。今回はフォトンマップ本に従ってRGBの反射率の平均を使う
				const double probability = (obj.color.x + obj.color.y + obj.color.z) / 3;
				if (probability > rand01()) { // 反射
					// orienting_normalの方向を基準とした正規直交基底(w, u, v)を作る。この基底に対する半球内で次のレイを飛ばす。
					Vec w, u, v;
					w = orienting_normal;
					if (fabs(w.x) > 0.1)
						u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
					else
						u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
					v = Cross(w, u);
					// コサイン項を使った重点的サンプリング
					const double r1 = 2 * PI * rand01();
					const double r2 = rand01(), r2s = sqrt(r2);
					Vec dir = Normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1.0 - r2)));

					now_ray = Ray(hitpoint, dir);
					now_flux = Multiply(now_flux, obj.color) / probability;
					continue;
				} else { // 吸収（すなわちここで追跡終了）
					trace_end = true;
					continue;
				}
			} break;
			}
		}
	}
	std::cout << "Done. (" << photon_map->Size() <<  " photons are stored)" << std::endl;
	std::cout << "Creating KD-tree..." << std::endl;
	photon_map->CreateKDtree();
	std::cout << "Done." << std::endl;
}

// 直接光を計算する
Color direct_radiance(const Vec &v0, const Vec &normal, const int id, const Vec &light_pos) {
	const Vec light_normal = Normalize(light_pos - spheres[LightID].position);
	const Vec light_dir = Normalize(light_pos - v0);
	const double dist2 = (light_pos - v0).LengthSquared();
	const double dot0 = Dot(normal, light_dir);
	const double dot1 = Dot(light_normal, -1.0 * light_dir);

	if (dot0 >= 0 && dot1 >= 0) {
		const double G = dot0 * dot1 / dist2;
		double t; // レイからシーンの交差位置までの距離
		int id_; // 交差したシーン内オブジェクトのID
		intersect_scene(Ray(v0, light_dir), &t, &id_);
		if (fabs(sqrt(dist2) - t) < 1e-3) {	
			return Multiply(spheres[id].color, spheres[LightID].emission) * (1.0 / PI) * G / (1.0 / (4.0 * PI * pow(spheres[LightID].radius, 2.0)));
		}
	}
	return Color();
}

// 光源上の点をサンプリングして直接光を計算する
Color direct_radiance_sample(const Vec &v0, const Vec &normal, const int id) {
	// 光源上の一点をサンプリングする
	const double r1 = 2 * PI * rand01();
	const double r2 = 1.0 - 2.0 * rand01();
	const Vec light_pos = spheres[LightID].position + ((spheres[LightID].radius + EPS) * Vec(sqrt(1.0 - r2*r2) * cos(r1), sqrt(1.0 - r2*r2) * sin(r1), r2));

	return direct_radiance(v0, normal, id, light_pos);
}

// ray方向からの放射輝度を求める
Color radiance(const Ray &ray, const int depth, PhotonMap *photon_map, IrradianceCache *cache, const double gather_radius, const int gahter_max_photon_num, const int final_gather, const int direct_light_samples, bool precompute = false) {
	double t; // レイからシーンの交差位置までの距離
	int id;   // 交差したシーン内オブジェクトのID
	if (!intersect_scene(ray, &t, &id))
		return BackgroundColor;

	const Sphere &obj = spheres[id];
	const Vec hitpoint = ray.org + t * ray.dir; // 交差位置
	const Vec normal  = Normalize(hitpoint - obj.position); // 交差位置の法線
	const Vec orienting_normal = Dot(normal, ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）

	// 色の反射率最大のものを得る。ロシアンルーレットで使う。
	// ロシアンルーレットの閾値は任意だが色の反射率等を使うとより良い。
	double russian_roulette_probability = std::max(obj.color.x, std::max(obj.color.y, obj.color.z));
	// 一定以上レイを追跡したらロシアンルーレットを実行し追跡を打ち切るかどうかを判断する
	if (depth > MaxDepth) {
		if (rand01() >= russian_roulette_probability)
			return obj.emission;
	} else
		russian_roulette_probability = 1.0; // ロシアンルーレット実行しなかった

	switch (obj.ref_type) {
	case DIFFUSE: {
		if (id == LightID)
			return obj.emission;
		// イラディアンスキャッシュをつかって放射輝度推定する
		IrradianceCache::ResultIrradianceQueue iqueue;
		const double threashold = 10.0; // 大きいほど精度向上
		const double Rmax = 100.0;
		const double radius = (1.0 / threashold) * Rmax;
		IrradianceCache::IrradianceQuery query(hitpoint, orienting_normal, radius * radius, threashold);
		cache->SearchCachedPoints(&iqueue, query);
		
		Color accum;
		if (iqueue.size() == 0 || precompute) { // 条件をみたすキャッシュ点見つからず
			// hitpointのイラディアンスを直接計算
			// 計算方法は何でもいい。今回はフォトンマップを使う
			double R0 = 0;

			// orienting_normalの方向を基準とした正規直交基底(w, u, v)を作る。この基底に対する半球内で次のレイを飛ばす。
			Vec w, u, v;
			w = orienting_normal;
			if (fabs(w.x) > 0.1)
				u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
			else
				u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
			v = Cross(w, u);
			
			// ファイナルギャザリング
			for (int fg = 0; fg < final_gather; fg ++) {
				// コサイン項を使った重点的サンプリング
				const double r1 = 2 * PI * rand01();
				const double r2 = rand01(), r2s = sqrt(r2);
				Vec dir = Normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1.0 - r2)));
				Ray ray_(hitpoint, dir);
				double t_; // レイからシーンの交差位置までの距離
				int id_;   // 交差したシーン内オブジェクトのID
				if (intersect_scene(ray_, &t_, &id_)) {
					R0 += 1.0 / t_;

					const Sphere &obj_ = spheres[id_];
					const Vec hitpoint_ = ray_.org + t_ * ray_.dir; // 交差位置
					const Vec normal_  = Normalize(hitpoint_ - obj_.position); // 交差位置の法線
					const Vec orienting_normal_ = Dot(normal_, ray_.dir) < 0.0 ? normal_ : (-1.0 * normal_); // 交差位置の法線（物体からのレイの入出を考慮）

					// フォトンマップをつかって放射輝度推定する
					PhotonMap::ResultQueue pqueue;
					// k近傍探索。gather_radius半径内のフォトンを最大gather_max_photon_num個集めてくる
					PhotonMap::Query query(hitpoint_, orienting_normal_, gather_radius, gahter_max_photon_num);
					photon_map->SearchKNN(&pqueue, query);
					Color accumulated_flux;
					double max_distance2 = -1;

					// キューからフォトンを取り出しvectorに格納する
					std::vector<PhotonMap::ElementForQueue> photons;
					photons.reserve(pqueue.size());
					for (;!pqueue.empty();) {
						PhotonMap::ElementForQueue p = pqueue.top(); pqueue.pop();
						photons.push_back(p);
						max_distance2 = std::max(max_distance2, p.distance2);
					}

					// 円錐フィルタを使用して放射輝度推定する
					const double max_distance = sqrt(max_distance2);
					const double k = 1.1;
					for (int i = 0; i < photons.size(); i ++) {
						const double weight = 1.0 - (sqrt(photons[i].distance2) / (k * max_distance)); // 円錐フィルタの重み
						const Color value = Multiply(obj_.color, photons[i].point->power) / PI; // Diffuse面のBRDF = 1.0 / πであったのでこれをかける
						accumulated_flux = accumulated_flux + weight * value;
					}
					accumulated_flux = accumulated_flux / (1.0 - 2.0 / (3.0 * k)); // 円錐フィルタの係数
					if (max_distance2 > 0.0) {
						accum = accum + accumulated_flux / (PI * max_distance2) / russian_roulette_probability / final_gather;
					}
				}
			}

			R0 = 1.0 / (R0 / final_gather);

			// イラディアンス追加
			cache->AddPointToTree(Irradiance(hitpoint, accum * PI, orienting_normal, R0));

			// イラディアンスからray方向への放射輝度推定
			accum = Multiply(obj.color, accum);
		} else {
			// イラディアンスキャッシュをつかって放射輝度推定する
			std::vector<IrradianceCache::ElementForIrradianceQueue> irrs;
			irrs.reserve(iqueue.size());
			double weight = 0;
			for (;!iqueue.empty();) {
				IrradianceCache::ElementForIrradianceQueue i = iqueue.top(); iqueue.pop();
				irrs.push_back(i);
				weight += i.weight;
			}

			for (int i = 0; i < irrs.size(); i ++) {
				accum = accum + irrs[i].weight * irrs[i].point->irradiance / weight / PI; // イラディアンス -> ラディアンスの変換のためにPIでわる（完全拡散面仮定）
			}
			// イラディアンスからray方向への放射輝度推定
			accum = Multiply(obj.color, accum);
		}
		 
		// 直接光のサンプリング
		// ファイナルギャザリングで求まるのは（今回は）間接光のみなので直接光は別にサンプリングして求めてやる
		for (int i = 0; i < direct_light_samples; i ++) {
			accum = accum + direct_radiance_sample(hitpoint, orienting_normal, id) / russian_roulette_probability / direct_light_samples;
		}

		return accum;
	} break;

	}

	return Color();
}

void pre_build_irradiance_cache(const Ray &camera, const Vec &cx, const Vec &cy, const int width, const int height,
	PhotonMap *photon_map, IrradianceCache *cache, const double gather_radius, const int gahter_max_photon_num, const int final_gather, const int direct_light_samples) {

	int x = rand01() * width; if (x >= width) x = width - 1;
	int y = rand01() * height; if (y >= height) y = height - 1;
	int sx = rand01() * 2; if (sx >= 2) sx = 1;
	int sy = rand01() * 2; if (sy >= 2) sy = 1;

	// テントフィルターによってサンプリング
	// ピクセル範囲で一様にサンプリングするのではなく、ピクセル中央付近にサンプルがたくさん集まるように偏りを生じさせる
	const double r1 = 2.0 * rand01(), dx = r1 < 1.0 ? sqrt(r1) - 1.0 : 1.0 - sqrt(2.0 - r1);
	const double r2 = 2.0 * rand01(), dy = r2 < 1.0 ? sqrt(r2) - 1.0 : 1.0 - sqrt(2.0 - r2);
	Vec dir = cx * (((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) +
				cy * (((sy + 0.5 + dy) / 2.0 + y) / height- 0.5) + camera.dir;
	radiance(Ray(camera.org + dir * 130.0, Normalize(dir)), 0, photon_map, cache, gather_radius, gahter_max_photon_num, final_gather, direct_light_samples, true);
}


// *** .hdrフォーマットで出力するための関数 ***
struct HDRPixel {
	unsigned char r, g, b, e;
	HDRPixel(const unsigned char r_ = 0, const unsigned char g_ = 0, const unsigned char b_ = 0, const unsigned char e_ = 0) :
	r(r_), g(g_), b(b_), e(e_) {};
	unsigned char get(int idx) {
		switch (idx) {
		case 0: return r;
		case 1: return g;
		case 2: return b;
		case 3: return e;
		} return 0;
	}

};

// doubleのRGB要素を.hdrフォーマット用に変換
HDRPixel get_hdr_pixel(const Color &color) {
	double d = std::max(color.x, std::max(color.y, color.z));
	if (d <= 1e-32)
		return HDRPixel();
	int e;
	double m = frexp(d, &e); // d = m * 2^e
	d = m * 256.0 / d;
	return HDRPixel(color.x * d, color.y * d, color.z * d, e + 128);
}

// 書き出し用関数
void save_hdr_file(const std::string &filename, const Color* image, const int width, const int height) {
	FILE *fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		std::cerr << "Error: " << filename << std::endl;
		return;
	}
	// .hdrフォーマットに従ってデータを書きだす
	// ヘッダ
	unsigned char ret = 0x0a;
	fprintf(fp, "#?RADIANCE%c", (unsigned char)ret);
	fprintf(fp, "# Made with 100%% pure HDR Shop%c", ret);
	fprintf(fp, "FORMAT=32-bit_rle_rgbe%c", ret);
	fprintf(fp, "EXPOSURE=1.0000000000000%c%c", ret, ret);

	// 輝度値書き出し
	fprintf(fp, "-Y %d +X %d%c", height, width, ret);
	for (int i = height - 1; i >= 0; i --) {
		std::vector<HDRPixel> line;
		for (int j = 0; j < width; j ++) {
			HDRPixel p = get_hdr_pixel(image[j + i * width]);
			line.push_back(p);
		}
		fprintf(fp, "%c%c", 0x02, 0x02);
		fprintf(fp, "%c%c", (width >> 8) & 0xFF, width & 0xFF);
		for (int i = 0; i < 4; i ++) {
			for (int cursor = 0; cursor < width;) {
				const int cursor_move = std::min(127, width - cursor);
				fprintf(fp, "%c", cursor_move);
				for (int j = cursor;  j < cursor + cursor_move; j ++)
					fprintf(fp, "%c", line[j].get(i));
				cursor += cursor_move;
			}
		}
	}

	fclose(fp);
}

int main(int argc, char **argv) {
	int width = 640;
	int height = 480;
	int photon_num = 50000;
	double gather_photon_radius = 32.0;
	int gahter_max_photon_num = 64;
	int final_gather = 64; // final_gather個のサンプル
	int direct_light_samples = 64;

	// カメラ位置
	Ray camera(Vec(50.0, 52.0, 295.6), Normalize(Vec(0.0, -0.042612, -1.0)));
	// シーン内でのスクリーンのx,y方向のベクトル
	Vec cx = Vec(width * 0.5135 / height);
	Vec cy = Normalize(Cross(cx, camera.dir)) * 0.5135;
	Color *image = new Color[width * height];

	// フォトンマップ構築
	PhotonMap photon_map;
	create_photon_map(photon_num, &photon_map);

	// イラディアンスキャッシュ
	// 事前構築（なにかと都合がよい）
	IrradianceCache cache;
	for (int i = 0; i < 10000; i ++) {
		if (i % 1000 == 0)
			std::cout << "*";
		pre_build_irradiance_cache(camera, cx, cy, width, height, &photon_map, &cache, gather_photon_radius, gahter_max_photon_num, final_gather, direct_light_samples);
	}

	for (int y = 0; y < height; y ++) {
		std::cerr << "Rendering " << (100.0 * y / (height - 1)) << "%" << std::endl;
		srand(y * y * y);
		for (int x = 0; x < width; x ++) {
			int image_index = y * width + x;	
			image[image_index] = Color();

			// 2x2のサブピクセルサンプリング
			for (int sy = 0; sy < 2; sy ++) {
				for (int sx = 0; sx < 2; sx ++) {
					// テントフィルターによってサンプリング
					// ピクセル範囲で一様にサンプリングするのではなく、ピクセル中央付近にサンプルがたくさん集まるように偏りを生じさせる
					const double r1 = 2.0 * rand01(), dx = r1 < 1.0 ? sqrt(r1) - 1.0 : 1.0 - sqrt(2.0 - r1);
					const double r2 = 2.0 * rand01(), dy = r2 < 1.0 ? sqrt(r2) - 1.0 : 1.0 - sqrt(2.0 - r2);
					Vec dir = cx * (((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) +
								cy * (((sy + 0.5 + dy) / 2.0 + y) / height- 0.5) + camera.dir;
					image[image_index] = image[image_index] + radiance(Ray(camera.org + dir * 130.0, Normalize(dir)), 0, &photon_map, &cache, gather_photon_radius, gahter_max_photon_num, final_gather, direct_light_samples);
				}
			}
		}
	}

	// .hdrフォーマットで出力
	save_hdr_file(std::string("image.hdr"), image, width, height);
}
