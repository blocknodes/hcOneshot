#!/usr/bin/env python3
"""1651 vod 批次中均分 <7 的 24 条：用影视域工具枚举多种工具/参数组合（探索候选，不调用任何服务）。

输出每条 query 的候选计划列表，计划格式与线上一致：[{"q","d":"vod","vt","params"}]。
参数写法沿用标注集惯例：vod_search_all 不写 retext（系统代填），vod_fuzzy_search 写 {}。
"""
import csv
import json
import sys

ALLOWED_FIELDS = {
    "title", "actor", "director", "dubbing", "entertainer", "prize", "language", "hostess", "role",
    "creation_source", "sound", "tag", "sub_prize", "target", "vender_name", "company", "writer",
    "definition", "area", "category", "channel", "comedy_brand", "gender", "technology",
    "is_fee", "is_over", "age_range", "release_time", "rate", "series", "video_index", "voice_start_pos",
}
RELATE_FIELDS = {"title", "actor", "director", "category"}
TOOLS = {"vod_search_all", "vod_fuzzy_search", "vod_relate_search", "vod_personalized_search",
         "vod_history", "vod_person_search", "fan_knowledge_agent"}


def L(field, value=None, values=None, op=None, frm=None, to=None):
    if frm is not None or to is not None:
        return {"field": field, "from": frm, "to": to}
    if values is not None:
        d = {"field": field, "values": values}
        if op:
            d["operator"] = op
        return d
    return {"field": field, "value": value}


def AND(*xs): return {"and": list(xs)}
def OR(*xs): return {"or": list(xs)}
def NOT(x): return {"not": x}


def S(query, action="search", sort=None):
    p = {"action": action, "query": query}
    if sort:
        p["sort"] = {k: {"order": v} for k, v in sort.items()}
    return ("vod_search_all", p)


def R(query): return ("vod_relate_search", {"query": query})
def F(): return ("vod_fuzzy_search", {})
def K(msg): return ("fan_knowledge_agent", {"messages": msg})
def P(cat=None): return ("vod_personalized_search", {"category": cat} if cat else {})
def H(cat=None, time=None):
    p = {}
    if cat:
        p["category"] = cat
    if time:
        p["time"] = time
    return ("vod_history", p)
def N(*names): return ("vod_person_search", {"person_name": list(names)})


# 每条：query -> [(说明, 工具, 参数) 或 多步 [(工具, 参数, 子句q)...]]
C = {
"豆瓣高分的老经典电影推荐": [
    ("评分≥8 + 电影，按评分降序", *S(AND(L("rate", frm="8.0", to="*"), L("category", "电影")), sort={"rate": "desc"})),
    ("评分≥8 + 电影 + 2000 年前上映", *S(AND(L("rate", frm="8.0", to="*"), L("category", "电影"), L("release_time", frm="*", to="19991231")), sort={"rate": "desc"})),
    ("评分≥8.5 + 电影 + 2010 年前上映", *S(AND(L("rate", frm="8.5", to="*"), L("category", "电影"), L("release_time", frm="*", to="20091231")), sort={"rate": "desc"})),
    ("标签经典 + 电影，按评分降序", *S(AND(L("tag", "经典"), L("category", "电影")), sort={"rate": "desc"})),
    ("标签经典 + 评分≥8 + 电影", *S(AND(L("tag", "经典"), L("rate", frm="8.0", to="*"), L("category", "电影")))),
    ("仅电影，按评分降序（不限年代）", *S(L("category", "电影"), sort={"rate": "desc"})),
    ("模糊检索（描述性推荐）", *F()),
    ("个性化推荐电影", *P("电影")),
],
"我这一辈子相声": [
    ("标题 + 起播", *S(L("title", "我这一辈子"), action="play")),
    ("标题 + 搜索", *S(L("title", "我这一辈子"))),
    ("标题 + 标签相声", *S(AND(L("title", "我这一辈子"), L("tag", "相声")), action="play")),
    ("标题 + 喜剧厂牌德云社", *S(AND(L("title", "我这一辈子"), L("comedy_brand", "德云社")), action="play")),
    ("标题 或 标签相声（宽召回）", *S(OR(L("title", "我这一辈子"), L("tag", "相声")))),
    ("模糊检索", *F()),
    ("知识问答：哪位演员说过", *K("我这一辈子相声")),
],
"请播放张云雷的相声": [
    ("演员 + 标签相声 + 起播", *S(AND(L("actor", "张云雷"), L("tag", "相声")), action="play")),
    ("演员 + 起播", *S(L("actor", "张云雷"), action="play")),
    ("演员 + 喜剧厂牌德云社 + 起播", *S(AND(L("actor", "张云雷"), L("comedy_brand", "德云社")), action="play")),
    ("参与人员 + 标签相声", *S(AND(L("entertainer", "张云雷"), L("tag", "相声")), action="play")),
    ("演员 或 参与人员 + 标签相声，按热度", *S(AND(OR(L("actor", "张云雷"), L("entertainer", "张云雷")), L("tag", "相声")), action="play", sort={"hot": "desc"})),
    ("演员 + 标签相声 + 搜索", *S(AND(L("actor", "张云雷"), L("tag", "相声")))),
    ("人物查询", *N("张云雷")),
    ("模糊检索", *F()),
],
"情侣吵架后看什么电影": [
    ("受众情侣 + 电影", *S(AND(L("target", "情侣"), L("category", "电影")))),
    ("受众情侣 + 标签爱情 + 电影", *S(AND(L("target", "情侣"), L("tag", "爱情"), L("category", "电影")))),
    ("标签爱情或喜剧 + 电影，按评分", *S(AND(L("tag", values=["爱情", "喜剧"]), L("category", "电影")), sort={"rate": "desc"})),
    ("标签治愈 + 电影", *S(AND(L("tag", "治愈"), L("category", "电影")))),
    ("标签爱情 + 电影 + 评分≥7.5", *S(AND(L("tag", "爱情"), L("category", "电影"), L("rate", frm="7.5", to="*")))),
    ("模糊检索（情绪场景）", *F()),
    ("个性化推荐电影", *P("电影")),
    ("知识问答（开放问题）", *K("情侣吵架后看什么电影")),
],
"推荐大人和孩子一起观看的影片": [
    ("受众家人 + 电影", *S(AND(L("target", "家人"), L("category", "电影")))),
    ("受众家人或儿童 + 电影", *S(AND(L("target", values=["家人", "儿童"]), L("category", "电影")))),
    ("标签合家欢 + 电影", *S(AND(L("tag", "合家欢"), L("category", "电影")))),
    ("标签合家欢或动画 + 电影，按评分", *S(AND(L("tag", values=["合家欢", "动画"]), L("category", "电影")), sort={"rate": "desc"})),
    ("受众家人（不限分类）", *S(L("target", "家人"))),
    ("受众家人 + 电影 + 评分≥7.5", *S(AND(L("target", "家人"), L("category", "电影"), L("rate", frm="7.5", to="*")))),
    ("模糊检索", *F()),
    ("个性化推荐电影", *P("电影")),
],
"我要看免费的霸天虎": [
    ("标题霸天虎 + 免费 + 起播", *S(AND(L("title", "霸天虎"), L("is_fee", "0")), action="play")),
    ("角色霸天虎 + 免费", *S(AND(L("role", "霸天虎"), L("is_fee", "0")), action="play")),
    ("角色或标题霸天虎 + 免费", *S(AND(OR(L("role", "霸天虎"), L("title", "霸天虎")), L("is_fee", "0")), action="play")),
    ("标题变形金刚 + 免费", *S(AND(L("title", "变形金刚"), L("is_fee", "0")), action="play")),
    ("标题霸天虎或变形金刚 + 免费", *S(AND(L("title", values=["霸天虎", "变形金刚"]), L("is_fee", "0")), action="play")),
    ("角色霸天虎（不限免费）", *S(L("role", "霸天虎"), action="play")),
    ("相关推荐：标题变形金刚", *R(L("title", "变形金刚"))),
    ("模糊检索", *F()),
],
"韩国恐怖片致命之旅": [
    ("标题 + 地区韩国 + 标签恐怖 + 电影", *S(AND(L("title", "致命之旅"), L("area", "韩国"), L("tag", "恐怖"), L("category", "电影")), action="play")),
    ("标题 + 地区韩国", *S(AND(L("title", "致命之旅"), L("area", "韩国")), action="play")),
    ("标题 + 标签恐怖", *S(AND(L("title", "致命之旅"), L("tag", "恐怖")), action="play")),
    ("标题 + 电影", *S(AND(L("title", "致命之旅"), L("category", "电影")), action="play")),
    ("仅标题", *S(L("title", "致命之旅"), action="play")),
    ("地区韩国 + 标签恐怖 + 电影（不带标题）", *S(AND(L("area", "韩国"), L("tag", "恐怖"), L("category", "电影")))),
    ("模糊检索", *F()),
],
"吕剧龙凤面": [
    ("标题 + 分类戏曲 + 起播", *S(AND(L("title", "龙凤面"), L("category", "戏曲")), action="play")),
    ("标题 + 标签吕剧", *S(AND(L("title", "龙凤面"), L("tag", "吕剧")), action="play")),
    ("标题 + 分类戏曲 + 标签吕剧", *S(AND(L("title", "龙凤面"), L("category", "戏曲"), L("tag", "吕剧")), action="play")),
    ("标题吕剧龙凤面（整串）", *S(L("title", "吕剧龙凤面"), action="play")),
    ("标题龙凤面或吕剧龙凤面 + 戏曲", *S(AND(L("title", values=["龙凤面", "吕剧龙凤面"]), L("category", "戏曲")), action="play")),
    ("仅标题龙凤面", *S(L("title", "龙凤面"), action="play")),
    ("标签吕剧 + 戏曲（不带标题）", *S(AND(L("tag", "吕剧"), L("category", "戏曲")))),
],
"普通话版的浪漫满屋": [
    ("标题 + 语言中文 + 电视剧", *S(AND(L("title", "浪漫满屋"), L("language", "中文"), L("category", "电视剧")), action="play")),
    ("标题 + 语言国语", *S(AND(L("title", "浪漫满屋"), L("language", "国语")), action="play")),
    ("标题 + 语言中文", *S(AND(L("title", "浪漫满屋"), L("language", "中文")), action="play")),
    ("标题 + 地区韩国 + 语言中文", *S(AND(L("title", "浪漫满屋"), L("area", "韩国"), L("language", "中文")), action="play")),
    ("标题 + 电视剧 + 地区韩国", *S(AND(L("title", "浪漫满屋"), L("category", "电视剧"), L("area", "韩国")), action="play")),
    ("标题 + 电视剧 + 排除综艺", *S(AND(L("title", "浪漫满屋"), L("category", "电视剧"), NOT(L("category", "综艺"))), action="play")),
    ("标题 + 配音版标签", *S(AND(L("title", "浪漫满屋"), L("tag", "国语配音")), action="play")),
],
"日语纪录片": [
    ("语言日语 + 纪录片", *S(AND(L("language", "日语"), L("category", "纪录片")))),
    ("地区日本 + 纪录片", *S(AND(L("area", "日本"), L("category", "纪录片")))),
    ("语言日语 + 地区日本 + 纪录片", *S(AND(L("language", "日语"), L("area", "日本"), L("category", "纪录片")))),
    ("（语言日语 或 地区日本）+ 纪录片", *S(AND(OR(L("language", "日语"), L("area", "日本")), L("category", "纪录片")))),
    ("语言日语 + 纪录片，按评分", *S(AND(L("language", "日语"), L("category", "纪录片")), sort={"rate": "desc"})),
    ("语言日语 + 纪录片，按热度", *S(AND(L("language", "日语"), L("category", "纪录片")), sort={"hot": "desc"})),
    ("语言日语 + 纪录片 + 起播", *S(AND(L("language", "日语"), L("category", "纪录片")), action="play")),
],
"开始播放浙江电视台的光荣之路": [
    ("标题 + 频道浙江卫视 + 起播", *S(AND(L("title", "光荣之路"), L("channel", "浙江卫视")), action="play")),
    ("标题 + 频道浙江卫视 + 电视剧", *S(AND(L("title", "光荣之路"), L("channel", "浙江卫视"), L("category", "电视剧")), action="play")),
    ("标题 + 频道（浙江卫视 或 浙江电视台）", *S(AND(L("title", "光荣之路"), L("channel", values=["浙江卫视", "浙江电视台"])), action="play")),
    ("标题 + 电视剧", *S(AND(L("title", "光荣之路"), L("category", "电视剧")), action="play")),
    ("标题 + 出品公司浙江卫视", *S(AND(L("title", "光荣之路"), L("company", "浙江卫视")), action="play")),
    ("仅标题 + 起播，按最新", *S(L("title", "光荣之路"), action="play", sort={"new": "desc"})),
    ("标题 + 排除 1957 年老片（2000 年后）", *S(AND(L("title", "光荣之路"), L("release_time", frm="20000101", to="*")), action="play")),
],
"快点我要看超级马里奥兄弟大电影": [
    ("标题全名 + 起播", *S(L("title", "超级马里奥兄弟大电影"), action="play")),
    ("标题全名 + 电影", *S(AND(L("title", "超级马里奥兄弟大电影"), L("category", "电影")), action="play")),
    ("标题超级马里奥兄弟 + 电影", *S(AND(L("title", "超级马里奥兄弟"), L("category", "电影")), action="play")),
    ("标题（全名 或 短名）+ 电影", *S(AND(L("title", values=["超级马里奥兄弟大电影", "超级马里奥兄弟"]), L("category", "电影")), action="play")),
    ("标题 + 电影 + 技术 CG 动画", *S(AND(L("title", "超级马里奥兄弟"), L("category", "电影"), L("technology", "cg动画")), action="play")),
    ("标题 + 电影 + 2023 年上映", *S(AND(L("title", "超级马里奥兄弟"), L("category", "电影"), L("release_time", frm="20230101", to="20231231")), action="play")),
    ("标题 + 电影 + 排除短视频", *S(AND(L("title", "超级马里奥兄弟"), L("category", "电影"), NOT(L("category", "短视频"))), action="play")),
],
"23年上映的影片": [
    ("2023 全年 + 电影", *S(AND(L("release_time", frm="20230101", to="20231231"), L("category", "电影")))),
    ("2023 全年 + 电影，按热度", *S(AND(L("release_time", frm="20230101", to="20231231"), L("category", "电影")), sort={"hot": "desc"})),
    ("2023 全年 + 电影，按评分", *S(AND(L("release_time", frm="20230101", to="20231231"), L("category", "电影")), sort={"rate": "desc"})),
    ("2023 全年 + 电影 + 排除短视频", *S(AND(L("release_time", frm="20230101", to="20231231"), L("category", "电影"), NOT(L("category", "短视频"))))),
    ("2023 全年 + 电影 + 正片", *S(AND(L("release_time", frm="20230101", to="20231231"), L("category", values=["电影", "正片"], op="and")))),
    ("2023 全年（不限分类）", *S(L("release_time", frm="20230101", to="20231231"))),
],
"刚上线的新剧里，哪个最值得追？": [
    ("近 3 个月 + 电视剧，按评分", *S(AND(L("release_time", frm="20260817", to="20261117"), L("category", "电视剧")), sort={"rate": "desc"})),
    ("近 1 个月 + 电视剧，按评分", *S(AND(L("release_time", frm="20261017", to="20261117"), L("category", "电视剧")), sort={"rate": "desc"})),
    ("近 3 个月 + 电视剧，按热度", *S(AND(L("release_time", frm="20260817", to="20261117"), L("category", "电视剧")), sort={"hot": "desc"})),
    ("电视剧 + 连载中，按评分", *S(AND(L("category", "电视剧"), L("is_over", "0")), sort={"rate": "desc"})),
    ("电视剧 + 连载中 + 近 3 个月，按热度", *S(AND(L("category", "电视剧"), L("is_over", "0"), L("release_time", frm="20260817", to="20261117")), sort={"hot": "desc"})),
    ("电视剧，按最新 + 评分", *S(L("category", "电视剧"), sort={"new": "desc", "rate": "desc"})),
    ("模糊检索", *F()),
    ("知识问答（开放问题）", *K("刚上线的新剧里，哪个最值得追？")),
],
"刚上线就播放量破千万的新剧": [
    ("近 3 个月 + 电视剧，按播放量", *S(AND(L("release_time", frm="20260817", to="20261117"), L("category", "电视剧")), sort={"play": "desc"})),
    ("近 1 个月 + 电视剧，按播放量", *S(AND(L("release_time", frm="20261017", to="20261117"), L("category", "电视剧")), sort={"play": "desc"})),
    ("近 3 个月 + 电视剧，按热度", *S(AND(L("release_time", frm="20260817", to="20261117"), L("category", "电视剧")), sort={"hot": "desc"})),
    ("电视剧 + 连载中，按播放量", *S(AND(L("category", "电视剧"), L("is_over", "0")), sort={"play": "desc"})),
    ("电视剧，按最新 + 播放量", *S(L("category", "电视剧"), sort={"new": "desc", "play": "desc"})),
    ("模糊检索", *F()),
    ("知识问答", *K("刚上线就播放量破千万的新剧")),
],
"播放不会员的诡娃娃": [
    ("标题 + 免费 + 起播", *S(AND(L("title", "诡娃娃"), L("is_fee", "0")), action="play")),
    ("标题（诡娃娃 或 鬼娃娃）+ 免费", *S(AND(L("title", values=["诡娃娃", "鬼娃娃"]), L("is_fee", "0")), action="play")),
    ("标题 + 免费 + 电影", *S(AND(L("title", "诡娃娃"), L("is_fee", "0"), L("category", "电影")), action="play")),
    ("标题 + 免费 + 标签恐怖", *S(AND(L("title", "诡娃娃"), L("is_fee", "0"), L("tag", "恐怖")), action="play")),
    ("标题 + 免费 + 排除短视频", *S(AND(L("title", "诡娃娃"), L("is_fee", "0"), NOT(L("category", "短视频"))), action="play")),
    ("仅标题 + 起播（不加免费）", *S(L("title", "诡娃娃"), action="play")),
],
"我想看和解密紫禁城类型相似的纪录片": [
    ("相关推荐：标题 + 纪录片", *R(AND(L("title", "解密紫禁城"), L("category", "纪录片")))),
    ("相关推荐：仅标题", *R(L("title", "解密紫禁城"))),
    ("检索：纪录片 + 标签历史", *S(AND(L("category", "纪录片"), L("tag", "历史")))),
    ("检索：纪录片 + 标签（历史 或 人文 或 文物）", *S(AND(L("category", "纪录片"), L("tag", values=["历史", "人文", "文物"])))),
    ("检索：纪录片 + 标签故宫", *S(AND(L("category", "纪录片"), L("tag", "故宫")))),
    ("检索：纪录片 + 标签历史 + 地区内地，按评分", *S(AND(L("category", "纪录片"), L("tag", "历史"), L("area", "内地")), sort={"rate": "desc"})),
    ("模糊检索", *F()),
],
"打开我昨天看的那部剧": [
    ("历史：电视剧 + 昨天", *H(["电视剧"], "2026-11-16 00:00:00 TO 2026-11-16 23:59:59")),
    ("历史：昨天（不限分类）", *H(None, "2026-11-16 00:00:00 TO 2026-11-16 23:59:59")),
    ("历史：电视剧（不限时间）", *H(["电视剧"])),
    ("历史：电视剧 + 昨天晚上", *H(["电视剧"], "2026-11-16 19:00:00 TO 2026-11-16 22:59:59")),
    ("历史：电视剧 + 近两天", *H(["电视剧"], "2026-11-15 00:00:00 TO 2026-11-16 23:59:59")),
    ("历史：电视剧 + 正片 + 昨天", *H(["电视剧", "正片"], "2026-11-16 00:00:00 TO 2026-11-16 23:59:59")),
],
"我想看豆瓣8.0评分以上的综艺": [
    ("评分≥8 + 综艺", *S(AND(L("rate", frm="8.0", to="*"), L("category", "综艺")))),
    ("评分≥8 + 综艺，按评分降序", *S(AND(L("rate", frm="8.0", to="*"), L("category", "综艺")), sort={"rate": "desc"})),
    ("评分≥8 + 综艺，按热度", *S(AND(L("rate", frm="8.0", to="*"), L("category", "综艺")), sort={"hot": "desc"})),
    ("评分≥8 + 综艺，按最新", *S(AND(L("rate", frm="8.0", to="*"), L("category", "综艺")), sort={"new": "desc"})),
    ("评分≥8 + 综艺 + 已完结", *S(AND(L("rate", frm="8.0", to="*"), L("category", "综艺"), L("is_over", "1")), sort={"rate": "desc"})),
    ("评分≥8 + 综艺 + 起播", *S(AND(L("rate", frm="8.0", to="*"), L("category", "综艺")), action="play", sort={"rate": "desc"})),
],
"有没有番茄台的中国梦之声 第1季": [
    ("标题 + 第 1 季 + 频道东方卫视", *S(AND(L("title", "中国梦之声"), L("series", "1"), L("channel", "东方卫视")))),
    ("标题 + 第 1 季", *S(AND(L("title", "中国梦之声"), L("series", "1")))),
    ("标题 + 第 1 季 + 综艺", *S(AND(L("title", "中国梦之声"), L("series", "1"), L("category", "综艺")))),
    ("标题（中国梦之声 或 中国梦之声第一季）", *S(L("title", values=["中国梦之声", "中国梦之声第一季"]))),
    ("标题 + 频道东方卫视", *S(AND(L("title", "中国梦之声"), L("channel", "东方卫视")))),
    ("标题 + 第 1 季 + 起播", *S(AND(L("title", "中国梦之声"), L("series", "1")), action="play")),
    ("标题 + 第 1 季 + 出品公司东方卫视", *S(AND(L("title", "中国梦之声"), L("series", "1"), L("company", "东方卫视")))),
],
"2020年之后拍的电影": [
    ("2020 起至今 + 电影", *S(AND(L("release_time", frm="20200101", to="*"), L("category", "电影")))),
    ("2020 起至今天 + 电影", *S(AND(L("release_time", frm="20200101", to="20261117"), L("category", "电影")))),
    ("2021 起（“之后”不含 2020）+ 电影", *S(AND(L("release_time", frm="20210101", to="*"), L("category", "电影")))),
    ("2020 起 + 电影，按热度", *S(AND(L("release_time", frm="20200101", to="*"), L("category", "电影")), sort={"hot": "desc"})),
    ("2020 起 + 电影，按评分", *S(AND(L("release_time", frm="20200101", to="*"), L("category", "电影")), sort={"rate": "desc"})),
    ("2020 起 + 电影，按最新", *S(AND(L("release_time", frm="20200101", to="*"), L("category", "电影")), sort={"new": "desc"})),
],
"跟半熟恋人相似的综艺有哪些": [
    ("相关推荐：标题 + 综艺", *R(AND(L("title", "半熟恋人"), L("category", "综艺")))),
    ("相关推荐：仅标题", *R(L("title", "半熟恋人"))),
    ("检索：综艺 + 标签恋爱", *S(AND(L("category", "综艺"), L("tag", "恋爱")))),
    ("检索：综艺 + 标签（恋爱 或 恋爱观察 或 情感）", *S(AND(L("category", "综艺"), L("tag", values=["恋爱", "恋爱观察", "情感"])))),
    ("检索：综艺 + 标签恋爱 + 受众情侣", *S(AND(L("category", "综艺"), L("tag", "恋爱"), L("target", "情侣")))),
    ("检索：综艺 + 标签恋爱，按热度", *S(AND(L("category", "综艺"), L("tag", "恋爱")), sort={"hot": "desc"})),
    ("知识问答", *K("跟半熟恋人相似的综艺有哪些")),
],
"我要看北京卫视的电视剧": [
    ("频道北京卫视 + 电视剧", *S(AND(L("channel", "北京卫视"), L("category", "电视剧")))),
    ("频道北京卫视 + 电视剧 + 起播", *S(AND(L("channel", "北京卫视"), L("category", "电视剧")), action="play")),
    ("频道北京卫视 + 电视剧，按最新", *S(AND(L("channel", "北京卫视"), L("category", "电视剧")), sort={"new": "desc"})),
    ("频道北京卫视 + 电视剧 + 连载中", *S(AND(L("channel", "北京卫视"), L("category", "电视剧"), L("is_over", "0")))),
    ("频道北京卫视 + 电视剧，按热度", *S(AND(L("channel", "北京卫视"), L("category", "电视剧")), sort={"hot": "desc"})),
    ("频道（北京卫视 或 BTV）+ 电视剧", *S(AND(L("channel", values=["北京卫视", "btv"]), L("category", "电视剧")))),
    ("出品公司北京卫视 + 电视剧（对照：线上现状）", *S(AND(L("company", "北京卫视"), L("category", "电视剧")))),
],
"黄渤被孩子气到发疯的电影": [
    ("模糊检索（剧情描述）", *F()),
    ("演员黄渤 + 电影", *S(AND(L("actor", "黄渤"), L("category", "电影")))),
    ("演员黄渤 + 电影 + 标签喜剧", *S(AND(L("actor", "黄渤"), L("category", "电影"), L("tag", "喜剧")))),
    ("演员黄渤 + 电影 + 标签（亲子 或 家庭）", *S(AND(L("actor", "黄渤"), L("category", "电影"), L("tag", values=["亲子", "家庭"])))),
    ("演员黄渤 + 电影 + 受众家人", *S(AND(L("actor", "黄渤"), L("category", "电影"), L("target", "家人")))),
    ("演员黄渤 + 电影 + 标签喜剧，按评分", *S(AND(L("actor", "黄渤"), L("category", "电影"), L("tag", "喜剧")), sort={"rate": "desc"})),
    ("知识问答：哪部电影", *K("黄渤被孩子气到发疯的电影")),
],
}


def walk(node, fields_ok, errs):
    if not isinstance(node, dict):
        errs.append("非对象节点"); return
    keys = set(node)
    if keys & {"and", "or"}:
        k = "and" if "and" in node else "or"
        if len(keys) != 1 or not isinstance(node[k], list) or not node[k]:
            errs.append(f"{k} 节点非法")
        for x in node.get(k, []):
            walk(x, fields_ok, errs)
    elif "not" in node:
        if len(keys) != 1:
            errs.append("not 节点非法")
        walk(node["not"], fields_ok, errs)
    else:
        f = node.get("field")
        if f not in fields_ok:
            errs.append(f"字段不允许:{f}")
        if ("value" in node) == ("values" in node) and "from" not in node:
            errs.append(f"{f} value/values 互斥")
        if "operator" in node and "values" not in node:
            errs.append(f"{f} operator 只能配 values")


def check(tool, p):
    errs = []
    if tool not in TOOLS:
        errs.append("未知工具")
    if tool == "vod_search_all":
        if p.get("action") not in ("search", "play"):
            errs.append("action 非法")
        walk(p.get("query"), ALLOWED_FIELDS, errs)
        for k in (p.get("sort") or {}):
            if k not in ("rate", "hot", "new", "play"):
                errs.append(f"sort 非法:{k}")
    elif tool == "vod_relate_search":
        walk(p.get("query"), RELATE_FIELDS, errs)
    elif tool == "vod_history":
        for c in p.get("category", []):
            if c not in ["电影", "电视剧", "纪录片", "综艺", "戏曲", "新闻", "资讯", "片花", "短视频", "广告", "正片", "动漫"]:
                errs.append(f"history 分类非法:{c}")
    elif tool == "vod_personalized_search":
        if p.get("category") not in (None, "电影", "电视剧", "纪录片", "综艺", "戏曲", "幼儿", "动漫"):
            errs.append("personalized 分类非法")
    return errs


def main(low_json, out_jsonl, out_csv):
    low = json.load(open(low_json))
    rows, n, bad = [], 0, 0
    with open(out_jsonl, "w") as w:
        for r in low:
            q = r["语料"]
            cands = []
            seen = set()
            for i, (desc, tool, params) in enumerate(C[q], 1):
                plan = [{"q": q, "d": "vod", "vt": f"vod:{tool}", "params": params}]
                key = json.dumps(plan, ensure_ascii=False, sort_keys=True)
                assert key not in seen, (q, desc)
                seen.add(key)
                errs = check(tool, params)
                bad += bool(errs)
                cands.append({"id": f"c{i}", "desc": desc, "tool": tool, "plan": plan, "schema_errors": errs})
                rows.append([r["用例ID"], q, f'{r["avg"]:.2f}', f"c{i}", tool, desc,
                             json.dumps(params, ensure_ascii=False, separators=(",", ":")), ";".join(errs)])
                n += 1
            w.write(json.dumps({"case_id": r["用例ID"], "query": q, "score_1651": round(r["avg"], 4),
                                "candidates": cands}, ensure_ascii=False) + "\n")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["用例ID", "query", "1651均分", "候选", "工具", "说明", "params", "schema问题"])
        cw.writerows(rows)
    print(f"query {len(low)} 条，候选 {n} 个，schema 问题 {bad} 个")


if __name__ == "__main__":
    main(*sys.argv[1:4])
