import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ValveConditionHomepage } from './valve-condition-homepage';

describe('ValveConditionHomepage', () => {
  let component: ValveConditionHomepage;
  let fixture: ComponentFixture<ValveConditionHomepage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ValveConditionHomepage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ValveConditionHomepage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
