"""Original configurable scenes. No third-party scene code or assets copied."""
import json
import os
from pathlib import Path
from manim import *
from animation_math import validate

SPEC = json.loads(Path(os.environ['ML_EXPLAIN_SCENE_SPEC']).read_text())
RESULT = validate(SPEC)
PALETTE = [BLUE_B, GREEN_B, YELLOW_B, PURPLE_B]


class TeachingScene(Scene):
    def setup(self):
        self.camera.background_color = '#111827'
        Text.set_default(font='Arial')
        title = Text(SPEC['title'], font_size=32)
        if title.width > 12:
            title.scale_to_fit_width(12)
        title.to_edge(UP, buff=.35)
        self.add(title)
        note = Text(SPEC.get('note', 'Illustrative values; presentation timing is not a measurement'), font_size=16, color=GRAY_B)
        if note.width > 12:
            note.scale_to_fit_width(12)
        note.to_edge(DOWN, buff=.25)
        self.add(note)

    def card(self, label, detail, shape, color):
        box = RoundedRectangle(width=8.3, height=2.9, corner_radius=.16, stroke_color=color, fill_color=color, fill_opacity=.08)
        lines = VGroup(Text(label, font_size=30, color=color), Text(shape, font_size=25), Text(detail, font_size=21)).arrange(DOWN, buff=.4)
        for line in lines:
            if line.width > 7.6:
                line.scale_to_fit_width(7.6)
        return VGroup(box, lines)


class TokenFlow(TeachingScene):
    def construct(self):
        stages = SPEC['stages']
        progress = VGroup(*[Dot(color=PALETTE[i % 4], radius=.09) for i in range(len(stages))]).arrange(RIGHT, buff=.6).move_to(DOWN*2.4)
        self.play(FadeIn(progress))
        card = self.card(**stages[0], color=PALETTE[0])
        cursor = Circle(radius=.19, color=WHITE).move_to(progress[0])
        self.play(FadeIn(card, shift=UP*.2), Create(cursor))
        self.wait(1.2)
        for i, stage in enumerate(stages[1:], 1):
            new = self.card(**stage, color=PALETTE[i % 4])
            self.play(ReplacementTransform(card, new), cursor.animate.move_to(progress[i]), run_time=1.2)
            card = new
            self.wait(1.4)
        self.wait(.8)


class CacheReuse(TeachingScene):
    def construct(self):
        blocks = SPEC['blocks']
        cards = VGroup()
        hashes = VGroup()
        pages = VGroup()
        n = len(blocks)
        for i, block in enumerate(blocks):
            x = (i-(n-1)/2)*3.8
            token = VGroup(RoundedRectangle(width=3.25,height=.8,corner_radius=.1,color=BLUE_B), Text(block['tokens'],font_size=22)).move_to([x,1.65,0])
            hashbox = VGroup(RoundedRectangle(width=3.25,height=.65,corner_radius=.1,color=YELLOW_B), Text(block['hash'],font_size=21,color=YELLOW_B)).move_to([x,.15,0])
            page = VGroup(RoundedRectangle(width=3.25,height=.75,corner_radius=.1,color=GRAY_B), Text(block['page']+' · reserved',font_size=18,color=GRAY_B)).move_to([x,-1.5,0])
            cards.add(token); hashes.add(hashbox); pages.add(page)
        state = Text('1. Split full blocks; hash depends on the previous block',font_size=22).move_to(UP*2.6)
        self.play(Write(state), LaggedStart(*[FadeIn(c) for c in cards],lag_ratio=.25))
        self.play(LaggedStart(*[TransformFromCopy(cards[i],hashes[i]) for i in range(n)],lag_ratio=.4))
        links = VGroup(*[Arrow(hashes[i-1].get_right(),hashes[i].get_left(),buff=.12,color=YELLOW_B) for i in range(1,n)])
        self.play(Create(links)); self.wait(1)
        new = Text('2. Register hash metadata and reserve pages; KV is not ready',font_size=22).move_to(state)
        self.play(Transform(state,new), FadeIn(pages)); self.wait(1.3)
        ready = VGroup()
        for i,block in enumerate(blocks):
            ready.add(VGroup(RoundedRectangle(width=3.25,height=.75,corner_radius=.1,color=GREEN_B,fill_color=GREEN_B,fill_opacity=.15),Text(block['page']+' · KV ready',font_size=18,color=GREEN_B)).move_to(pages[i]))
        new = Text('3. Model execution completes; pages now hold usable KV',font_size=22).move_to(state)
        self.play(Transform(state,new),ReplacementTransform(pages,ready),run_time=1.5); self.wait(1)
        new = Text('4. A later request reuses the completed prefix',font_size=22).move_to(state)
        reuse = VGroup(*[CurvedArrow(cards[i].get_right(),ready[i].get_right(),angle=-PI/3,color=GREEN_B) for i in range(n)])
        self.play(Transform(state,new),LaggedStart(*[Create(a) for a in reuse],lag_ratio=.3),run_time=1.5)
        self.wait(2)


class AttentionComputation(TeachingScene):
    def construct(self):
        def matrix(rows):
            return Matrix([[r'-\infty' if x is None else f'{x:.2f}' for x in row] for row in rows], element_to_mobject=MathTex, h_buff=1.15, v_buff=.7).scale(.7)
        def show(label, rows, color):
            title = MathTex(label,color=color,font_size=38).move_to(UP*2.1)
            mat = matrix(rows).move_to(DOWN*.1)
            if mat.width > 11:
                mat.scale_to_fit_width(11)
            return VGroup(title,mat)
        inputs = VGroup(*[VGroup(MathTex(name,font_size=34,color=PALETTE[i]),matrix(SPEC[key])).arrange(DOWN,buff=.4) for i,(name,key) in enumerate([('Q','q'),('K','k'),('V','v')])]).arrange(RIGHT,buff=.65)
        if inputs.width > 12:
            inputs.scale_to_fit_width(12)
        self.play(FadeIn(inputs)); self.wait(1.7)
        current = show(r'S=QK^T/\sqrt{d_k}',RESULT['scores'],BLUE_B)
        self.play(ReplacementTransform(inputs,current),run_time=1.5); self.wait(1.4)
        steps = [(r'S+M\quad\text{(causal mask)}' if SPEC.get('causal',True) else r'S\quad\text{(no mask)}', 'masked_scores',YELLOW_B), (r'A=\operatorname{softmax}_{\rm row}(S+M)' if SPEC.get('causal',True) else r'A=\operatorname{softmax}_{\rm row}(S)', 'weights',PURPLE_B), (r'O=AV', 'output',GREEN_B)]
        for label,key,color in steps:
            new = show(label,RESULT[key],color)
            self.play(ReplacementTransform(current,new),run_time=1.4)
            current = new
            self.wait(1.7)
        self.wait(1)
